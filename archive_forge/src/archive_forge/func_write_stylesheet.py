import os
import warnings
from os import path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from docutils.frontend import OptionParser
from docutils.nodes import Node
import sphinx.builders.latex.nodes  # NOQA  # Workaround: import this before writer to avoid ImportError
from sphinx import addnodes, highlighting, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.builders.latex.constants import ADDITIONAL_SETTINGS, DEFAULT_SETTINGS, SHORTHANDOFF
from sphinx.builders.latex.theming import Theme, ThemeFactory
from sphinx.builders.latex.util import ExtBabel
from sphinx.config import ENUM, Config
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import NoUri, SphinxError
from sphinx.locale import _, __
from sphinx.util import logging, progress_message, status_iterator, texescape
from sphinx.util.console import bold, darkgreen  # type: ignore
from sphinx.util.docutils import SphinxFileOutput, new_document
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.nodes import inline_all_toctrees
from sphinx.util.osutil import SEP, make_filename_from_project
from sphinx.util.template import LaTeXRenderer
from sphinx.writers.latex import LaTeXTranslator, LaTeXWriter
from docutils import nodes  # isort:skip
def write_stylesheet(self) -> None:
    highlighter = highlighting.PygmentsBridge('latex', self.config.pygments_style)
    stylesheet = path.join(self.outdir, 'sphinxhighlight.sty')
    with open(stylesheet, 'w', encoding='utf-8') as f:
        f.write('\\NeedsTeXFormat{LaTeX2e}[1995/12/01]\n')
        f.write('\\ProvidesPackage{sphinxhighlight}[2022/06/30 stylesheet for highlighting with pygments]\n')
        f.write('% Its contents depend on pygments_style configuration variable.\n\n')
        f.write(highlighter.get_stylesheet())