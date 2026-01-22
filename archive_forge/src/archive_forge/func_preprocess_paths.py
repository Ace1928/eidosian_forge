from send2trash.compat import text_type, binary_type, iterable_type
def preprocess_paths(paths):
    if isinstance(paths, iterable_type) and (not isinstance(paths, (text_type, binary_type))):
        paths = list(paths)
    elif not isinstance(paths, list):
        paths = [paths]
    paths = [path.__fspath__() if hasattr(path, '__fspath__') else path for path in paths]
    return paths