import re
import types
from IPython.utils.dir2 import dir2
def filter_ns(ns, name_pattern='*', type_pattern='all', ignore_case=True, show_all=True):
    """Filter a namespace dictionary by name pattern and item type."""
    pattern = name_pattern.replace('*', '.*').replace('?', '.')
    if ignore_case:
        reg = re.compile(pattern + '$', re.I)
    else:
        reg = re.compile(pattern + '$')
    return dict(((key, obj) for key, obj in ns.items() if reg.match(key) and show_hidden(key, show_all) and is_type(obj, type_pattern)))