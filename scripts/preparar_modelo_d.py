"""Reconstruye los arrays de D a partir de los manifiestos publicados y NIH ChestX-ray14."""

import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from rx_torax.data.checksums import read_manifest, sha256
from rx_torax.data.splits import NAMES
from rx_torax.prepare import write_json
from rx_torax.preprocessing.images import transform


CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
MANIFESTS = CONFIGS / 'nih_effusion_pa_modelo_b_v1_manifiestos.zip'
CHECKSUMS = CONFIGS / 'nih_effusion_pa_modelo_b_v1.sha256'
ROUNDING = CONFIGS / 'nih_effusion_pa_modelo_b_v1_redondeo.npz'


def fill_arrays(nih, destination, rows, size):
    remaining = {}
    with ExitStack() as resources:
        arrays = {}
        for split, items in rows.items():
            indices = [int(row['array_index']) for row in items]
            if indices != list(range(len(items))):
                raise ValueError(f'Los índices de {split} están desordenados')
            for row in items:
                name = row['image_path']
                if name in remaining:
                    raise ValueError(f'Imagen repetida en los manifiestos: {name}')
                remaining[name] = (split, int(row['array_index']), row['pixel_hash'])
            arrays[split] = np.lib.format.open_memmap(destination / f'X_{split}.npy', mode='w+',
                                                     dtype=np.float32, shape=(len(items), size, size))
            resources.callback(arrays[split]._mmap.close)
            np.save(destination / f'y_{split}.npy', np.array([int(row['target']) for row in items], dtype=np.int64),
                    allow_pickle=False)
        filled = set()
        for number in range(1, 13):
            archive_path = nih / 'images' / f'images_{number:03}.tar.gz'
            print(f'Reconstruyendo desde {archive_path.name} ({number}/12)', flush=True)
            with tarfile.open(archive_path, mode='r|gz', bufsize=1024 * 1024) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    name = PurePosixPath(member.name).name
                    if name in filled:
                        raise ValueError(f'Imagen repetida en los archivos NIH: {name}')
                    if name in remaining:
                        split, index, expected = remaining.pop(name)
                        with archive.extractfile(member) as source:
                            try:
                                arrays[split][index] = transform(io.BytesIO(source.read()), size, expected)
                            except ValueError as error:
                                raise ValueError(f'{name}: {error}') from error
                        filled.add(name)
        if remaining:
            raise ValueError(f'Faltan {len(remaining)} imágenes; por ejemplo {next(iter(remaining))}')
        for array in arrays.values():
            array.flush()


def verify_arrays(directory):
    """Repite el redondeo original de Pillow en los píxeles donde varía entre plataformas."""
    expected = read_manifest(CHECKSUMS)
    with np.load(ROUNDING, allow_pickle=False) as reference:
        for split in NAMES:
            image_file = directory / f'X_{split}.npy'
            if sha256(image_file) != expected[image_file.name]:
                array = np.lib.format.open_memmap(image_file, mode='r+')
                try:
                    indices, values = reference[f'{split}_indices'], reference[f'{split}_values']
                    pixels = array.reshape(-1)
                    if np.any(np.abs(pixels[indices] - values) > 1e-6):
                        raise ValueError(f'{image_file.name}: diferencia superior a la tolerancia de redondeo')
                    pixels[indices] = values
                    array.flush()
                    del pixels
                finally:
                    array._mmap.close()
                if sha256(image_file) != expected[image_file.name]:
                    raise ValueError(f'{image_file.name} difiere del array original')
            labels_file = directory / f'y_{split}.npy'
            if sha256(labels_file) != expected[labels_file.name]:
                raise ValueError(f'{labels_file.name} difiere del array original')


def prepare(nih, output):
    if output.exists():
        raise ValueError(f'La salida ya existe: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=output.name + '.preparing-', dir=output.parent) as folder:
        stage = Path(folder)
        with ZipFile(MANIFESTS) as source:
            for name in [*(f'{split}.csv' for split in NAMES), 'exclusiones.csv', 'labels.json',
                         'preprocessing.json', 'config.json', 'report.json']:
                (stage / name).write_bytes(source.read(name))
        metadata = nih / 'Data_Entry_2017_v2020.csv'
        report = json.loads((stage / 'report.json').read_text(encoding='utf-8'))
        if hashlib.sha256(metadata.read_bytes()).hexdigest() != report['metadata_sha256']:
            raise ValueError('El CSV de NIH difiere del que se usó para preparar D')
        rows = {}
        for split in NAMES:
            with (stage / f'{split}.csv').open(newline='', encoding='utf-8') as stream:
                rows[split] = list(csv.DictReader(stream))
        size = json.loads((stage / 'preprocessing.json').read_text(encoding='utf-8'))['image_size']
        required = sum(len(items) for items in rows.values()) * size * size * np.dtype(np.float32).itemsize
        if shutil.disk_usage(stage).free < required * 1.1:
            raise ValueError('Espacio insuficiente para reconstruir los arrays de D')
        config = json.loads((stage / 'config.json').read_text(encoding='utf-8'))
        config.update(metadata=str(metadata), image_root=str(nih / 'images'), output=str(output))
        write_json(stage / 'config.json', config)
        fill_arrays(nih, stage, rows, size)
        print('Verificando los seis arrays contra el SHA-256 original...', flush=True)
        verify_arrays(stage)
        stage.rename(output)
    print(f'Conjunto de D reconstruido: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nih', type=Path, required=True)
    parser.add_argument('--salida', type=Path, default=Path('data/processed/nih_effusion_pa_modelo_b_v1'))
    args = parser.parse_args()
    try:
        prepare(args.nih.resolve(), args.salida.resolve())
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
