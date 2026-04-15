import matplotlib
matplotlib.use('Agg')
import json, traceback, sys

with open('notebooks/reman_im_sim.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

ns = {'__name__': '__main__'}
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']).strip()
    if not src or 'ipywidgets' in src or 'display(' in src:
        sys.stdout.write(f'[SKIP] cell {i}\n')
        sys.stdout.flush()
        continue
    src = src.replace('plt.show()', 'plt.close("all")')
    src = src.replace('plt.savefig', '# plt.savefig')
    try:
        exec(compile(src, f'cell-{i}', 'exec'), ns)
        sys.stdout.write(f'[OK]    cell {i}\n')
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f'[ERROR] cell {i}: {type(e).__name__}: {e}\n')
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        break

sys.stdout.write('Done.\n')
