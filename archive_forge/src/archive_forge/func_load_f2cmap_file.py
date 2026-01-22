from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def load_f2cmap_file(f2cmap_file):
    global f2cmap_all, f2cmap_mapped
    f2cmap_all = copy.deepcopy(f2cmap_default)
    if f2cmap_file is None:
        f2cmap_file = '.f2py_f2cmap'
        if not os.path.isfile(f2cmap_file):
            return
    try:
        outmess('Reading f2cmap from {!r} ...\n'.format(f2cmap_file))
        with open(f2cmap_file) as f:
            d = eval(f.read().lower(), {}, {})
        f2cmap_all, f2cmap_mapped = process_f2cmap_dict(f2cmap_all, d, c2py_map, True)
        outmess('Successfully applied user defined f2cmap changes\n')
    except Exception as msg:
        errmess('Failed to apply user defined f2cmap changes: %s. Skipping.\n' % msg)