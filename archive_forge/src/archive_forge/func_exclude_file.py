from . import VENDORED_ROOT
from ._util import cwd, iter_all_files
def exclude_file(dirname, basename):
    if dirname == 'pydevd':
        if basename in INCLUDES:
            return False
        elif not basename.endswith('.py'):
            return True
        elif 'pydev' not in basename:
            return True
        return False
    if basename.endswith('.pyc'):
        return True
    return False