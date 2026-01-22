def get_dir_with_slash(path):
    if path == b'' or path.endswith(b'/'):
        return path
    else:
        dirname, basename = posixpath.split(path)
        if dirname == b'':
            return dirname
        else:
            return dirname + b'/'