from pathlib import Path
from numpy.lib._iotools import _is_string_like
def get_file_obj(fname, mode='r', encoding=None):
    """
    Light wrapper to handle strings, path objects and let files (anything else)
    pass through.

    It also handle '.gz' files.

    Parameters
    ----------
    fname : str, path object or file-like object
        File to open / forward
    mode : str
        Argument passed to the 'open' or 'gzip.open' function
    encoding : str
        For Python 3 only, specify the encoding of the file

    Returns
    -------
    A file-like object that is always a context-manager. If the `fname` was
    already a file-like object, the returned context manager *will not
    close the file*.
    """
    if _is_string_like(fname):
        fname = Path(fname)
    if isinstance(fname, Path):
        return fname.open(mode=mode, encoding=encoding)
    elif hasattr(fname, 'open'):
        return fname.open(mode=mode, encoding=encoding)
    try:
        return open(fname, mode, encoding=encoding)
    except TypeError:
        try:
            if 'r' in mode:
                fname.read
            if 'w' in mode or 'a' in mode:
                fname.write
        except AttributeError:
            raise ValueError('fname must be a string or a file-like object')
        return EmptyContextManager(fname)