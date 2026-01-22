import sys
import os
from _pydev_bundle._pydev_execfile import execfile
def runfile(filename, args=None, wdir=None, namespace=None):
    """
    Run filename
    args: command line arguments (string)
    wdir: working directory
    """
    try:
        if hasattr(filename, 'decode'):
            filename = filename.decode('utf-8')
    except (UnicodeError, TypeError):
        pass
    global __umd__
    if os.environ.get('PYDEV_UMD_ENABLED', '').lower() == 'true':
        if __umd__ is None:
            namelist = os.environ.get('PYDEV_UMD_NAMELIST', None)
            if namelist is not None:
                namelist = namelist.split(',')
            __umd__ = UserModuleDeleter(namelist=namelist)
        else:
            verbose = os.environ.get('PYDEV_UMD_VERBOSE', '').lower() == 'true'
            __umd__.run(verbose=verbose)
    if args is not None and (not isinstance(args, (bytes, str))):
        raise TypeError('expected a character buffer object')
    if namespace is None:
        namespace = _get_globals()
    if '__file__' in namespace:
        old_file = namespace['__file__']
    else:
        old_file = None
    namespace['__file__'] = filename
    sys.argv = [filename]
    if args is not None:
        for arg in args.split():
            sys.argv.append(arg)
    if wdir is not None:
        try:
            if hasattr(wdir, 'decode'):
                wdir = wdir.decode('utf-8')
        except (UnicodeError, TypeError):
            pass
        os.chdir(wdir)
    execfile(filename, namespace)
    sys.argv = ['']
    if old_file is None:
        del namespace['__file__']
    else:
        namespace['__file__'] = old_file