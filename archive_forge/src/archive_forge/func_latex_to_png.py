from io import BytesIO, open
import os
import tempfile
import shutil
import subprocess
from base64 import encodebytes
import textwrap
from pathlib import Path
from IPython.utils.process import find_cmd, FindCmdError
from traitlets.config import get_config
from traitlets.config.configurable import SingletonConfigurable
from traitlets import List, Bool, Unicode
from IPython.utils.py3compat import cast_unicode
def latex_to_png(s, encode=False, backend=None, wrap=False, color='Black', scale=1.0):
    """Render a LaTeX string to PNG.

    Parameters
    ----------
    s : str
        The raw string containing valid inline LaTeX.
    encode : bool, optional
        Should the PNG data base64 encoded to make it JSON'able.
    backend : {matplotlib, dvipng}
        Backend for producing PNG data.
    wrap : bool
        If true, Automatically wrap `s` as a LaTeX equation.
    color : string
        Foreground color name among dvipsnames, e.g. 'Maroon' or on hex RGB
        format, e.g. '#AA20FA'.
    scale : float
        Scale factor for the resulting PNG.
    None is returned when the backend cannot be used.

    """
    s = cast_unicode(s)
    allowed_backends = LaTeXTool.instance().backends
    if backend is None:
        backend = allowed_backends[0]
    if backend not in allowed_backends:
        return None
    if backend == 'matplotlib':
        f = latex_to_png_mpl
    elif backend == 'dvipng':
        f = latex_to_png_dvipng
        if color.startswith('#'):
            if len(color) == 7:
                try:
                    color = 'RGB {}'.format(' '.join([str(int(x, 16)) for x in textwrap.wrap(color[1:], 2)]))
                except ValueError as e:
                    raise ValueError('Invalid color specification {}.'.format(color)) from e
            else:
                raise ValueError('Invalid color specification {}.'.format(color))
    else:
        raise ValueError('No such backend {0}'.format(backend))
    bin_data = f(s, wrap, color, scale)
    if encode and bin_data:
        bin_data = encodebytes(bin_data)
    return bin_data