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
Render the LaTeX math expression *math* using latex and dvipng or
    dvisvgm.

    Return the filename relative to the built document and the "depth",
    that is, the distance of image bottom and baseline in pixels, if the
    option to use preview_latex is switched on.
    Also return the temporary and destination files.

    Error handling may seem strange, but follows a pattern: if LaTeX or dvipng
    (dvisvgm) aren't available, only a warning is generated (since that enables
    people on machines without these programs to at least build the rest of the
    docs successfully).  If the programs are there, however, they may not fail
    since that indicates a problem in the math source.
    