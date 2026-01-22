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
def genelatex(body, wrap):
    """Generate LaTeX document for dvipng backend."""
    lt = LaTeXTool.instance()
    breqn = wrap and lt.use_breqn and kpsewhich('breqn.sty')
    yield '\\documentclass{article}'
    packages = lt.packages
    if breqn:
        packages = packages + ['breqn']
    for pack in packages:
        yield '\\usepackage{{{0}}}'.format(pack)
    yield '\\pagestyle{empty}'
    if lt.preamble:
        yield lt.preamble
    yield '\\begin{document}'
    if breqn:
        yield '\\begin{dmath*}'
        yield body
        yield '\\end{dmath*}'
    elif wrap:
        yield u'$${0}$$'.format(body)
    else:
        yield body
    yield u'\\end{document}'