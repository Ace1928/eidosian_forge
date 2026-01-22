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
def convert_dvi_to_svg(dvipath: str, builder: Builder, out_path: str) -> Optional[int]:
    """Convert DVI file to SVG image."""
    name = 'dvisvgm'
    command = [builder.config.imgmath_dvisvgm, '-o', out_path]
    command.extend(builder.config.imgmath_dvisvgm_args)
    command.append(dvipath)
    stdout, stderr = convert_dvi_to_image(command, name)
    depth = None
    if builder.config.imgmath_use_preview:
        for line in stderr.splitlines():
            matched = depthsvg_re.match(line)
            if matched:
                depth = round(float(matched.group(1)) * 100 / 72.27)
                write_svg_depth(out_path, depth)
                break
    return depth