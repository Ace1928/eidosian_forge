from pathlib import Path
import numpy as np
from ase.io import read, write
def test_xsf_spec():
    files = {'01-comments': f1, '02-atoms': f2, '03-periodic': f3, '04-forces-atoms': f4, '05-forces-slab': f5, '06-anim-atoms': f6, '07-anim-crystal-fixcell': f7, '08-anim-crystal-varcell': f8}
    names = list(sorted(files.keys()))
    for name in names:
        check(name, files[name], check_data=False)
        check('%s-ignore-datagrid' % name, files[name] + datagrid, check_data=False)
        check('%s-read-datagrid' % name, files[name] + datagrid, check_data=True)