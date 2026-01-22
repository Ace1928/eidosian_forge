import stat
from ... import controldir
def mode_to_kind(mode):
    if mode in (420, 33188):
        return ('file', False)
    elif mode in (493, 33261):
        return ('file', True)
    elif mode == 16384:
        return ('directory', False)
    elif mode == 40960:
        return ('symlink', False)
    elif mode == 57344:
        return ('tree-reference', False)
    else:
        raise AssertionError('invalid mode %o' % mode)