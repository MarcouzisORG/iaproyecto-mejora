"""Prepara una variante del conjunto de D reemplazando negativas dentro de cada paciente."""

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from rx_torax.data.splits import negative_kind
from rx_torax.prepare import write_csv, write_json
from rx_torax.preprocessing.images import read_image, transform


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def image_name(path):
    return path.replace('\\', '/').rsplit('/', 1)[-1]


def eligible_replacements(nih, splits, exclusions, metadata_rows):
    metadata = {row['Image Index']: row for row in metadata_rows}
    eligible_images = set((nih / 'train_val_list.txt').read_text(encoding='utf-8-sig').splitlines())
    official_test = (nih / 'test_list.txt').read_text(encoding='utf-8-sig').splitlines()
    held_out = {metadata[name]['Patient ID'] for name in official_test}
    held_out.update(row['patient_id'] for name in ('val', 'test') for row in splits[name])
    used = {image_name(row['image_path']) for rows in splits.values() for row in rows}
    used.update(image_name(row['image_path']) for row in exclusions)

    slots = defaultdict(list)
    for row in splits['train']:
        name = image_name(row['image_path'])
        visits = int(metadata[name]['Follow-up #'])
        if row['target'] == '0' and visits < 10:
            slots[row['patient_id'], negative_kind(row)].append((visits, int(row['array_index']), row))

    candidates = defaultdict(list)
    for number, row in enumerate(metadata_rows, start=2):
        name, patient = row['Image Index'], row['Patient ID']
        findings = row['Finding Labels']
        kind = negative_kind({'target_original': findings})
        visits = int(row['Follow-up #'])
        if ((patient, kind) in slots and patient not in held_out and name in eligible_images and name not in used
                and row['View Position'] == 'PA' and 'Effusion' not in findings.split('|') and visits >= 10):
            candidates[patient, kind].append((visits, name, number, findings))
    return slots, candidates


def audit_archives(archives, wanted, working, baseline_hashes):
    """Extrae solo candidatos, verifica los píxeles y registra de qué archivo salieron."""
    found, rejected = {}, Counter()
    duplicate_names = set()
    for index, path in enumerate(archives, start=1):
        print(f'Auditando archivo {index}/{len(archives)}: {path.name}', flush=True)
        with tarfile.open(path, mode='r|gz', bufsize=1024 * 1024) as archive:
            for member in archive:
                name = image_name(member.name)
                if name not in wanted or not member.isfile():
                    continue
                if name in found or name in duplicate_names:
                    duplicate_names.add(name)
                    continue
                output = working / name
                with archive.extractfile(member) as source, output.open('wb') as target:
                    shutil.copyfileobj(source, target)
                try:
                    _, info = read_image(output)
                except (OSError, ValueError, Warning):
                    info = None
                found[name] = (path.name, info)

    fingerprints = Counter(info['pixel_hash'] for _, info in found.values() if info is not None)
    accepted = {}
    for name in sorted(wanted):
        if name not in found:
            rejected['missing'] += 1
            continue
        source, info = found[name]
        if name in duplicate_names:
            rejected['duplicate_name'] += 1
        elif info is None:
            rejected['unreadable'] += 1
        elif info['pixel_hash'] in baseline_hashes or fingerprints[info['pixel_hash']] > 1:
            rejected['duplicate_pixels'] += 1
        else:
            accepted[name] = (source, info)
    return accepted, dict(rejected)


def prepare(base, nih, output):
    if output.exists():
        raise ValueError(f'La salida ya existe: {output}')
    metadata_path = nih / 'Data_Entry_2017_v2020.csv'
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    original = json.loads((base / 'report.json').read_text(encoding='utf-8'))
    if metadata_hash != original['metadata_sha256']:
        raise ValueError('El CSV de NIH debe coincidir con el usado para preparar el conjunto base')
    splits = {name: read_rows(base / f'{name}.csv') for name in ('train', 'val', 'test')}
    exclusions = read_rows(base / 'exclusiones.csv')
    train_rows = splits['train']
    manifest_labels = [int(row['target']) for row in train_rows]
    if ([int(row['array_index']) for row in train_rows] != list(range(len(train_rows)))
            or manifest_labels != np.load(base / 'y_train.npy', allow_pickle=False).tolist()):
        raise ValueError('train.csv no coincide con y_train.npy')
    preprocessing = json.loads((base / 'preprocessing.json').read_text(encoding='utf-8'))
    if preprocessing['normalization'] != 'unit':
        raise ValueError('El conjunto base debe estar preparado con normalización unit')

    slots, candidates = eligible_replacements(nih, splits, exclusions, read_rows(metadata_path))
    wanted = {name for rows in candidates.values() for _, name, _, _ in rows}
    if not wanted:
        raise ValueError('No hay imágenes candidatas para los reemplazos')
    baseline_hashes = {row['pixel_hash'] for rows in splits.values() for row in rows}
    archives = [nih / 'images' / f'images_{index:03}.tar.gz' for index in range(1, 13)]
    output.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = sum((base / f'X_{name}.npy').stat().st_size for name in splits)
    if shutil.disk_usage(output.parent).free < source_bytes * 2:
        raise ValueError('Espacio insuficiente para la copia de los arrays y la auditoría')

    with TemporaryDirectory(prefix=output.name + '.preparing-', dir=output.parent) as location:
        stage = Path(location)
        with TemporaryDirectory(prefix='candidatas-', dir=stage) as raw_images:
            working = Path(raw_images)
            accepted, rejected = audit_archives(archives, wanted, working, baseline_hashes)
            matches = []
            for key, rows in candidates.items():
                available = sorted(candidate for candidate in rows if candidate[1] in accepted)
                matches.extend(zip(sorted(slots[key]), available))
            if not matches:
                raise ValueError('La auditoría no dejó imágenes válidas para reemplazar')
            print(f'{len(matches)} reemplazos válidos. Copiando los arrays del conjunto base...', flush=True)
            for name in splits:
                shutil.copyfile(base / f'X_{name}.npy', stage / f'X_{name}.npy')
                shutil.copyfile(base / f'y_{name}.npy', stage / f'y_{name}.npy')
                if name != 'train':
                    shutil.copyfile(base / f'{name}.csv', stage / f'{name}.csv')
            shutil.copyfile(base / 'labels.json', stage / 'labels.json')
            shutil.copyfile(base / 'exclusiones.csv', stage / 'exclusiones.csv')
            arrays = np.lib.format.open_memmap(stage / 'X_train.npy', mode='r+')
            changes = []
            try:
                size = preprocessing['image_size']
                if arrays.shape != (len(train_rows), size, size) or arrays.dtype != np.float32:
                    raise ValueError('X_train.npy no tiene la forma float32 esperada')
                sources = stage / 'fuentes_nuevas'
                sources.mkdir()
                for (before, index, row), (after, name, number, findings) in sorted(matches):
                    source, info = accepted[name]
                    file = working / name
                    arrays[index] = transform(file, size, info['pixel_hash'])
                    file.replace(sources / name)
                    train_rows[index] = dict(row, **info, source_row=number, target_original=findings,
                                             image_path=str(output / sources.name / name))
                    changes.append({'array_index': index, 'patient_id': row['patient_id'],
                                    'original': image_name(row['image_path']), 'replacement': name,
                                    'follow_up_before': before, 'follow_up_after': after,
                                    'negative_kind': negative_kind(row), 'archive': source,
                                    'pixel_hash': info['pixel_hash']})
                arrays.flush()
            finally:
                arrays._mmap.close()
                del arrays
        write_csv(stage / 'train.csv', train_rows)
        write_csv(stage / 'reemplazos.csv', changes)
        config = json.loads((base / 'config.json').read_text(encoding='utf-8'))
        config.update(metadata=str(metadata_path), image_root=str(nih / 'images'), output=str(output))
        write_json(stage / 'config.json', config)
        preprocessing['statistics_source'] = 'modelo_d_train_after_resize_and_padding'
        write_json(stage / 'preprocessing.json', preprocessing)
        write_json(stage / 'report.json', {
            'base': str(base), 'metadata_sha256': metadata_hash, 'candidates': len(wanted),
            'rejected': rejected, 'replacements': len(changes),
            'replacement_patients': len({row['patient_id'] for row in changes}),
            'replacements_by_negative_kind': dict(Counter(row['negative_kind'] for row in changes)),
            'splits': original['splits'],
        })
        stage.rename(output)
    print(f'Conjunto preparado: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--nih', type=Path, required=True)
    parser.add_argument('--salida', type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.base.resolve(), args.nih.resolve(), args.salida.resolve())
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
