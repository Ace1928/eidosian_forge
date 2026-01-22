import base64
import re
import shutil
import subprocess
import tempfile
from os import path
from subprocess import PIPE, CalledProcessError
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Element
import sphinx
from sphinx import package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging, sha1
from sphinx.util.math import get_node_equation_number, wrap_displaymath
from sphinx.util.osutil import ensuredir
from sphinx.util.png import read_png_depth, write_png_depth
from sphinx.util.template import LaTeXRenderer
from sphinx.writers.html import HTMLTranslator
def clean_up_files(app: Sphinx, exc: Exception) -> None:
    if exc:
        return
    if hasattr(app.builder, '_imgmath_tempdir'):
        try:
            shutil.rmtree(app.builder._imgmath_tempdir)
        except Exception:
            pass
    if app.builder.config.imgmath_embed:
        try:
            shutil.rmtree(path.join(app.builder.outdir, app.builder.imagedir, 'math'))
        except Exception:
            pass