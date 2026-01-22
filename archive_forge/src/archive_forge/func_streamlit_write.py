from __future__ import annotations
import contextlib
import errno
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
@contextlib.contextmanager
def streamlit_write(path, binary=False):
    """Opens a file for writing within the streamlit path, and
    ensuring that the path exists. For example:

        with streamlit_write('foo/bar.txt') as bar:
            ...

    opens the file .streamlit/foo/bar.txt for writing,
    creating any necessary directories along the way.

    path   - the path to write to (within the streamlit directory)
    binary - set to True for binary IO
    """
    mode = 'w'
    if binary:
        mode += 'b'
    path = get_streamlit_file_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, mode) as handle:
            yield handle
    except OSError as e:
        msg = ['Unable to write file: %s' % os.path.abspath(path)]
        if e.errno == errno.EINVAL and env_util.IS_DARWIN:
            msg.append('Python is limited to files below 2GB on OSX. See https://bugs.python.org/issue24658')
        raise util.Error('\n'.join(msg))