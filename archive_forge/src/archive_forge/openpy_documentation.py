import io
from io import TextIOWrapper, BytesIO
from pathlib import Path
import re
from tokenize import open, detect_encoding
Read a Python file from a URL, using the encoding declared inside the file.

    Parameters
    ----------
    url : str
        The URL from which to fetch the file.
    errors : str
        How to handle decoding errors in the file. Options are the same as for
        bytes.decode(), but here 'replace' is the default.
    skip_encoding_cookie : bool
        If True (the default), and the encoding declaration is found in the first
        two lines, that line will be excluded from the output.

    Returns
    -------
    A unicode string containing the contents of the file.
    