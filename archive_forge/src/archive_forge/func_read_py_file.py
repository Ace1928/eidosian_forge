import io
from io import TextIOWrapper, BytesIO
from pathlib import Path
import re
from tokenize import open, detect_encoding
def read_py_file(filename, skip_encoding_cookie=True):
    """Read a Python file, using the encoding declared inside the file.

    Parameters
    ----------
    filename : str
        The path to the file to read.
    skip_encoding_cookie : bool
        If True (the default), and the encoding declaration is found in the first
        two lines, that line will be excluded from the output.

    Returns
    -------
    A unicode string containing the contents of the file.
    """
    filepath = Path(filename)
    with open(filepath) as f:
        if skip_encoding_cookie:
            return ''.join(strip_encoding_cookie(f))
        else:
            return f.read()