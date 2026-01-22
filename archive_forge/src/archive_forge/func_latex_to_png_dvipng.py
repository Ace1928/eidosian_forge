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
def latex_to_png_dvipng(s, wrap, color='Black', scale=1.0):
    try:
        find_cmd('latex')
        find_cmd('dvipng')
    except FindCmdError:
        return None
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        workdir = Path(tempfile.mkdtemp())
        tmpfile = 'tmp.tex'
        dvifile = 'tmp.dvi'
        outfile = 'tmp.png'
        with workdir.joinpath(tmpfile).open('w', encoding='utf8') as f:
            f.writelines(genelatex(s, wrap))
        subprocess.check_call(['latex', '-halt-on-error', '-interaction', 'batchmode', tmpfile], cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
        resolution = round(150 * scale)
        subprocess.check_call(['dvipng', '-T', 'tight', '-D', str(resolution), '-z', '9', '-bg', 'Transparent', '-o', outfile, dvifile, '-fg', color], cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
        with workdir.joinpath(outfile).open('rb') as f:
            return f.read()
    except subprocess.CalledProcessError:
        return None
    finally:
        shutil.rmtree(workdir)