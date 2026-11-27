"""CLI: rx-entrenar --config configs/modelo_a.json [--salida models/otra_carpeta]. Usa train y val; nunca abre test."""

import argparse
import csv
import importlib.metadata
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rx_torax.data.splits import FOLLOW_UP_BINS, follow_up_bin, quotas
from rx_torax.evaluation.metrics import best_threshold, binary_metrics, roc_auc, trivial_baselines
from rx_torax.training.model import build_model, to_input


DEFAULTS = {'architecture': 'densenet121', 'pretrained': True, 'epochs': 15, 'patience': 4, 'batch_size': 32,
            'learning_rate': 1e-4, 'weight_decay': 1e-4, 'augmentation': True, 'repeat_positives': 1, 'seed': 42,
            'device': 'auto', 'fine_tune_last_block': False, 'selection_metric': 'auc',
            'other_findings_negative_weight': 1.0, 'nih_follow_up_metadata': None}


def load_config(path, output=None):
    """Lee el JSON de entrenamiento; output reemplaza la carpeta de salida del JSON."""
    with Path(path).open(encoding='utf-8-sig') as stream:
        given = json.load(stream)
    if output is not None:
        given['output'] = str(output)
    required = {'data', 'output'}
    if not required <= given.keys() or given.keys() - required - set(DEFAULTS):
        raise ValueError(f"Configuración: faltan {sorted(required-given.keys())}; "
                         f"claves desconocidas {sorted(given.keys()-required-set(DEFAULTS))}")
    config = DEFAULTS | given
    metadata = config['nih_follow_up_metadata']
    if metadata is not None and (not isinstance(metadata, str) or not metadata.strip()):
        raise ValueError('nih_follow_up_metadata debe ser una ruta al CSV de NIH')
    weight = config['other_findings_negative_weight']
    if type(weight) not in (int, float) or weight < 1 or not math.isfinite(weight):
        raise ValueError('other_findings_negative_weight debe ser un número finito mayor o igual a 1')
    if config['selection_metric'] not in ('auc', 'f1'):
        raise ValueError('selection_metric debe ser auc o f1')
    if not isinstance(config['fine_tune_last_block'], bool):
        raise ValueError('fine_tune_last_block debe ser booleano')
    if config['fine_tune_last_block'] and config['architecture'] != 'efficientnet_b0':
        raise ValueError('El ajuste del último bloque está disponible para efficientnet_b0')
    for name in ['epochs', 'patience', 'batch_size', 'repeat_positives']:
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f'{name} debe ser un entero positivo')
    if Path(config['output']).exists():
        raise ValueError(f"La salida ya existe: {config['output']}. Usar otra ruta para conservar el experimento")
    return config


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_split(data, name, limit=None):
    images = np.load(data / f'X_{name}.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(data / f'y_{name}.npy', allow_pickle=False)
    indices = np.arange(len(labels))
    if limit and limit < len(labels):
        indices = np.linspace(0, len(labels) - 1, limit, dtype=int)
    return images, labels, indices


def repeat_positive_indices(indices, labels, repeat):
    """Cada positivo aparece `repeat` veces por época. Solo se usa con train."""
    return np.concatenate([indices, np.repeat(indices[labels[indices] == 1], repeat - 1)])


def read_training_rows(path, labels, fields):
    """Lee el manifiesto en el mismo orden que los arrays preparados."""
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = (fields | {'array_index', 'target'}) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: faltan columnas {sorted(missing)}')
        rows = list(reader)
    if [int(row['array_index']) for row in rows] != list(range(len(labels))) or \
            [int(row['target']) for row in rows] != labels.tolist():
        raise ValueError(f'{path}: las filas no coinciden con y_train.npy')
    return rows


def other_findings_weights(path, labels, weight):
    """Identifica las negativas con otros hallazgos usando solo la metadata de train."""
    rows = read_training_rows(path, labels, {'target_original'})
    findings = np.array([row['target_original'] for row in rows])
    return np.where((labels == 0) & (findings != 'No Finding'), weight, 1.0).astype(np.float32)


def load_train_follow_up(path, metadata, labels):
    rows = read_training_rows(path, labels, {'image_path'})
    # Los manifiestos preparados pueden traer rutas de otra plataforma.
    names = [PurePosixPath(row['image_path'].replace('\\', '/')).name for row in rows]
    wanted = set(names)
    with metadata.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'Image Index', 'Follow-up #'} <= set(reader.fieldnames or []):
            raise ValueError(f'{metadata}: se necesitan Image Index y Follow-up #')
        groups = {row['Image Index']: follow_up_bin(int(row['Follow-up #']))
                  for row in reader if row['Image Index'] in wanted}
    if missing := wanted - groups.keys():
        raise ValueError(f'{metadata}: faltan {len(missing)} imágenes de train')
    return np.array([groups[name] for name in names])


def match_follow_up_negatives(indices, labels, follow_up, rng):
    """Iguala la mezcla de seguimientos de ambas clases manteniendo sus tamaños por época."""
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    if not len(positive) or not len(negative):
        raise ValueError('El muestreo por estudios previos necesita ambas clases en train')
    shares = {group: int(np.count_nonzero(follow_up[positive] == group)) for group in FOLLOW_UP_BINS}
    counts = quotas(len(negative), shares)
    chosen = [positive]
    for group, count in counts.items():
        if count == 0:
            continue
        candidates = negative[follow_up[negative] == group]
        if not len(candidates):
            raise ValueError(f'Faltan negativas con {group} estudios previos para balancear train')
        # Repartir las repeticiones por igual y sortear solo el resto de la cuota.
        repeats, remainder = divmod(count, len(candidates))
        chosen.extend([np.tile(candidates, repeats), rng.choice(candidates, size=remainder, replace=False)])
    return np.concatenate(chosen)


def augment(batch, generator):
    """Rotación, escala y desplazamiento leves por imagen, sin espejar: el tórax no es simétrico."""
    count = len(batch)
    uniform = lambda limit: (torch.rand(count, generator=generator) * 2 - 1) * limit
    angle = uniform(7 * math.pi / 180)
    scale = 1 + uniform(0.05)
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0] = theta[:, 1, 1] = torch.cos(angle) / scale
    theta[:, 0, 1] = -torch.sin(angle) / scale
    theta[:, 1, 0] = torch.sin(angle) / scale
    theta[:, 0, 2] = uniform(0.1)
    theta[:, 1, 2] = uniform(0.1)
    images = batch.unsqueeze(1)
    grid = F.affine_grid(theta.to(batch.device), images.size(), align_corners=False)
    return F.grid_sample(images, grid, padding_mode='zeros', align_corners=False).squeeze(1)


def batch_of(images, indices, device):
    return torch.from_numpy(np.array(images[indices], dtype=np.float32)).to(device)


def predict_logits(model, images, indices, mean, std, batch_size, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = batch_of(images, indices[start:start + batch_size], device)
            logits.append(model(to_input(batch, mean, std)).squeeze(1).cpu())
    return torch.cat(logits)


def train_epoch(model, images, labels, indices, mean, std, optimizer, config, device, rng, generator,
                *, sample_weights=None):
    model.train()
    if config.get('fine_tune_last_block', False):
        for block in model.features:
            # Los bloques congelados conservan las estadísticas de BatchNorm.
            block.train(any(parameter.requires_grad for parameter in block.parameters()))
    order = rng.permutation(indices)
    total = 0.0
    for start in range(0, len(order), config['batch_size']):
        chosen = order[start:start + config['batch_size']]
        batch = batch_of(images, chosen, device)
        if config['augmentation']:
            batch = augment(batch, generator)
        target = torch.from_numpy(labels[chosen].astype(np.float32)).to(device)
        optimizer.zero_grad()
        logits = model(to_input(batch, mean, std)).squeeze(1)
        if sample_weights is None:
            loss = F.binary_cross_entropy_with_logits(logits, target)
        else:
            weights = torch.from_numpy(sample_weights[chosen]).to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target, weight=weights, reduction='sum') / weights.sum()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(chosen)
    return total / len(order)


def versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy')}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(config, limit=None):
    data = Path(config['data'])
    output = Path(config['output'])
    stage = output.with_name(output.name + '.parcial')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    preprocessing = json.loads((data / 'preprocessing.json').read_text(encoding='utf-8'))
    target_name = json.loads((data / 'labels.json').read_text(encoding='utf-8'))['target']
    mean, std = preprocessing['train_mean'], preprocessing['train_std']
    images, labels, unique_indices = load_split(data, 'train', limit)
    base_indices = repeat_positive_indices(unique_indices, labels, config['repeat_positives'])
    metadata = config.get('nih_follow_up_metadata')
    follow_up = load_train_follow_up(data / 'train.csv', Path(metadata), labels) if metadata else None
    weight = config.get('other_findings_negative_weight', DEFAULTS['other_findings_negative_weight'])
    epoch_options = {}
    if weight != 1:
        epoch_options['sample_weights'] = other_findings_weights(data / 'train.csv', labels, weight)
    val_images, val_labels, val_indices = load_split(data, 'val', limit)
    device = pick_device(config['device'])
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    sampling_rng = np.random.default_rng([config['seed'], 1])
    generator = torch.Generator().manual_seed(config['seed'])
    model = build_model(config['architecture'], config['pretrained']).to(device)
    fine_tune_last_block = config.get('fine_tune_last_block', False)
    if fine_tune_last_block:
        model.features.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=1)
    labels_val = val_labels[val_indices]
    val_target = torch.from_numpy(labels_val.astype(np.float32))
    selection_metric = config.get('selection_metric', DEFAULTS['selection_metric'])
    print(f'Entrenando {config["architecture"]} en {device}: {len(unique_indices)} imágenes de train '
          f'({len(base_indices)} por época), {len(val_indices)} de val')

    checkpoint = stage / 'modelo.pt'
    history, best_value, best_epoch = [], -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        indices = base_indices
        if follow_up is not None:
            indices = match_follow_up_negatives(base_indices, labels, follow_up, sampling_rng)
        if fine_tune_last_block and epoch == 2:
            # Última etapa MBConv y proyección final de EfficientNet-B0.
            model.features[-2:].requires_grad_(True)
        epoch_started = time.perf_counter()
        rate = optimizer.param_groups[0]['lr']
        train_loss = train_epoch(model, images, labels, indices, mean, std, optimizer, config, device, rng, generator,
                                 **epoch_options)
        logits = predict_logits(model, val_images, val_indices, mean, std, config['batch_size'] * 2, device)
        val_loss = F.binary_cross_entropy_with_logits(logits, val_target).item()
        val_auc = roc_auc(labels_val, logits.numpy())
        scores = torch.sigmoid(logits).numpy().astype(np.float64)
        threshold = best_threshold(labels_val, scores)
        val_f1 = binary_metrics(labels_val, scores, threshold)['f1']
        scheduler.step(val_auc)
        selection_value = val_f1 if selection_metric == 'f1' else val_auc
        history.append({'epoch': epoch, 'learning_rate': rate, 'train_loss': train_loss, 'val_loss': val_loss,
                        'val_auc': val_auc, 'val_f1': val_f1, 'seconds': time.perf_counter() - epoch_started})
        write_csv(stage / 'historial.csv', history)
        print(f'Época {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} '
              f'val_auc={val_auc:.4f} val_f1={val_f1:.4f} '
              f'({history[-1]["seconds"]:.0f} s)')
        if selection_value > best_value:
            best_value, best_epoch = selection_value, epoch
            torch.save({'architecture': config['architecture'], 'state_dict': model.state_dict(), 'train_mean': mean,
                        'train_std': std, 'image_size': preprocessing['image_size'], 'target': target_name,
                        'epoch': epoch}, checkpoint)
        elif epoch - best_epoch >= config['patience']:
            print(f'Sin mejora de {selection_metric.upper()} en {config["patience"]} épocas; '
                  'se detiene el entrenamiento')
            break

    # Las métricas salen del artefacto guardado, no del último estado en memoria.
    artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(artifact['state_dict'])
    logits = predict_logits(model, val_images, val_indices, mean, std, config['batch_size'] * 2, device)
    scores = torch.sigmoid(logits).numpy().astype(np.float64)
    threshold = best_threshold(labels_val, scores)
    artifact['threshold'] = threshold
    torch.save(artifact, checkpoint)
    write_csv(stage / 'predicciones_val.csv',
              [{'array_index': int(index), 'target': int(label), 'score': float(score)}
               for index, label, score in zip(val_indices, labels_val, scores)])
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'config': config, 'limit': limit,
              'device': str(device), 'versions': versions(), 'train_images': len(unique_indices),
              'train_samples_per_epoch': len(indices), 'train_positive_fraction': float(labels[indices].mean()),
              'val_images': len(val_indices), 'best_epoch': best_epoch, 'epochs_run': len(history),
              'train_seconds': time.perf_counter() - started,
              'val': {'auc': roc_auc(labels_val, scores), **binary_metrics(labels_val, scores, threshold)},
              'val_threshold_0_5': binary_metrics(labels_val, scores, 0.5),
              'val_baselines': trivial_baselines(labels_val),
              'note': 'La época y el umbral se eligieron con val, así que estas métricas son optimistas. Test no se usó.'}
    if follow_up is not None:
        report['train_follow_up_counts_per_epoch'] = {
            name: {group: int(np.count_nonzero((labels[indices] == target) & (follow_up[indices] == group)))
                   for group in FOLLOW_UP_BINS}
            for name, target in [('positives', 1), ('negatives', 0)]
        }
    (stage / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--salida', type=Path, help='Carpeta de salida en lugar de la del JSON (no debe existir)')
    parser.add_argument('--limit', type=int, help='Usa como máximo N imágenes por conjunto, para probar el flujo')
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.salida)
    except (ValueError, OSError) as error:
        parser.exit(2, f'ERROR: {error}\n')
    report = train(config, args.limit)
    print(json.dumps({'best_epoch': report['best_epoch'], 'val': report['val']}, indent=2))


if __name__ == '__main__':
    main()
