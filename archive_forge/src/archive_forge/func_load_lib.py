import os
import sys
import ctypes
def load_lib(exact_lib_names, lib_names, lib_dirs=None):
    """load_lib(exact_lib_names, lib_names, lib_dirs=None)

    Load a dynamic library.

    This function first tries to load the library from the given exact
    names. When that fails, it tries to find the library in common
    locations. It searches for files that start with one of the names
    given in lib_names (case insensitive). The search is performed in
    the given lib_dirs and a set of common library dirs.

    Returns ``(ctypes_library, library_path)``
    """
    assert isinstance(exact_lib_names, list)
    assert isinstance(lib_names, list)
    if lib_dirs is not None:
        assert isinstance(lib_dirs, list)
    exact_lib_names = [n for n in exact_lib_names if n]
    lib_names = [n for n in lib_names if n]
    if lib_names:
        the_lib_name = lib_names[0]
    elif exact_lib_names:
        the_lib_name = exact_lib_names[0]
    else:
        raise ValueError('No library name given.')
    if SYSTEM_LIBS_ONLY:
        lib_dirs, lib_paths = ([], [])
    else:
        lib_dirs, lib_paths = generate_candidate_libs(lib_names, lib_dirs)
    lib_paths = exact_lib_names + lib_paths
    if sys.platform.startswith('win'):
        loader = ctypes.windll
    else:
        loader = ctypes.cdll
    the_lib = None
    errors = []
    for fname in lib_paths:
        try:
            the_lib = loader.LoadLibrary(fname)
            break
        except Exception as err:
            if fname not in exact_lib_names:
                errors.append((fname, err))
    if the_lib is None:
        if errors:
            err_txt = ['%s:\n%s' % (lib, str(e)) for lib, e in errors]
            msg = 'One or more %s libraries were found, but ' + 'could not be loaded due to the following errors:\n%s'
            raise OSError(msg % (the_lib_name, '\n\n'.join(err_txt)))
        else:
            msg = 'Could not find a %s library in any of:\n%s'
            raise OSError(msg % (the_lib_name, '\n'.join(lib_dirs)))
    return (the_lib, fname)