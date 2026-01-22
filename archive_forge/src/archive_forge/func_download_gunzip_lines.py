import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def download_gunzip_lines(remote):
    """Downloads a file from a remote location and gunzips it.

    Returns the lines in the file."""
    import gzip
    from urllib.request import urlopen
    with urlopen(remote) as zfd:
        with gzip.open(zfd, mode='rt') as gfd:
            return gfd.readlines()