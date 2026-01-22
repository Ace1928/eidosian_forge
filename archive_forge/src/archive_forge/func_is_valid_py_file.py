def is_valid_py_file(path):
    """
    Checks whether the file can be read by the coverage module. This is especially
    needed for .pyx files and .py files with syntax errors.
    """
    import os
    is_valid = False
    if os.path.isfile(path) and (not os.path.splitext(path)[1] == '.pyx'):
        try:
            with open(path, 'rb') as f:
                compile(f.read(), path, 'exec')
                is_valid = True
        except:
            pass
    return is_valid