import os
def get_extension_libs():
    """Return the .c files in the `numba.cext` directory.
    """
    libs = []
    base = get_path()
    for fn in os.listdir(base):
        if fn.endswith('.c'):
            fn = os.path.join(base, fn)
            libs.append(fn)
    return libs