"""Execute notebooks in fresh kernels using this Python environment."""

import argparse
import base64
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for name, directory in {
    'MPLCONFIGDIR': ROOT / '.cache' / 'matplotlib',
    'IPYTHONDIR': ROOT / '.cache' / 'ipython',
    'JUPYTER_CONFIG_DIR': ROOT / '.jupyter',
    'JUPYTER_RUNTIME_DIR': ROOT / '.jupyter' / 'runtime',
}.items():
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, str(directory))

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def execute(path, timeout=600):
    path = path.resolve()
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop('execution', None)

    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']

    def progress(cell, cell_index):
        if cell.cell_type == 'code' and cell.source.strip():
            print(f'{path.name}: cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

    client = NotebookClient(
        notebook,
        km=manager,
        resources={'metadata': {'path': str(path.parent)}},
        timeout=timeout,
        allow_errors=False,
        force_raise_errors=True,
        on_cell_start=progress,
    )
    started = time.monotonic()
    try:
        client.execute(cleanup_kc=True)
    except Exception:
        failure_dir = ROOT / 'outputs'
        failure_dir.mkdir(exist_ok=True)
        nbformat.write(notebook, failure_dir / f'{path.stem}.failed.ipynb')
        raise

    code_cells = [c for c in notebook.cells if c.cell_type == 'code' and c.source.strip()]
    if any(c.execution_count is None for c in code_cells):
        raise RuntimeError(f'{path.name}: some code cells were not executed')
    if any(o.output_type == 'error' for c in code_cells for o in c.outputs):
        raise RuntimeError(f'{path.name}: error output found')
    plotly_charts = [
        output
        for cell in code_cells
        for output in cell.outputs
        if 'application/vnd.plotly.v1+json' in output.get('data', {})
    ]
    for chart in plotly_charts:
        preview = chart['data'].get('image/png', '')
        if not base64.b64decode(preview).startswith(b'\x89PNG\r\n\x1a\n'):
            raise RuntimeError(f'{path.name}: a Plotly chart has no valid PNG preview for GitHub')
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(
        f'PASS {path.name}: {len(code_cells)} code cells, '
        f'{len(plotly_charts)} Plotly charts with PNG previews, {time.monotonic() - started:.1f}s',
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', type=Path)
    parser.add_argument('--timeout', type=int, default=600, help='Timeout per cell in seconds')
    args = parser.parse_args()
    paths = args.notebooks or [ROOT / 'DBA_project_ForecastabilityIndex.ipynb', ROOT / 'setup/DBA_project_setup.ipynb']
    for path in paths:
        execute(path, args.timeout)


if __name__ == '__main__':
    main()
