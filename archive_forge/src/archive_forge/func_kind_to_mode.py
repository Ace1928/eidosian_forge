import stat
from ... import controldir
def kind_to_mode(kind, executable):
    if kind == 'file':
        if executable is True:
            return stat.S_IFREG | 493
        elif executable is False:
            return stat.S_IFREG | 420
        else:
            raise AssertionError('Executable %r invalid' % executable)
    elif kind == 'symlink':
        return stat.S_IFLNK
    elif kind == 'directory':
        return stat.S_IFDIR
    elif kind == 'tree-reference':
        return 57344
    else:
        raise AssertionError("Unknown file kind '%s'" % kind)