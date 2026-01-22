def common_path(path1, path2):
    """Find the common bit of 2 paths."""
    return b''.join(_common_path_and_rest(path1, path2)[0])