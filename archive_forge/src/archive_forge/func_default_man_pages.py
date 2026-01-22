import warnings
from os import path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from docutils.frontend import OptionParser
from docutils.io import FileOutput
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.locale import __
from sphinx.util import logging, progress_message
from sphinx.util.console import darkgreen  # type: ignore
from sphinx.util.nodes import inline_all_toctrees
from sphinx.util.osutil import ensuredir, make_filename_from_project
from sphinx.writers.manpage import ManualPageTranslator, ManualPageWriter
def default_man_pages(config: Config) -> List[Tuple[str, str, str, List[str], int]]:
    """ Better default man_pages settings. """
    filename = make_filename_from_project(config.project)
    return [(config.root_doc, filename, '%s %s' % (config.project, config.release), [config.author], 1)]