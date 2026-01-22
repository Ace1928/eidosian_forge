import os
import warnings
from os import path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from docutils import nodes
from docutils.frontend import OptionParser
from docutils.io import FileOutput
from docutils.nodes import Node
from sphinx import addnodes, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.util import logging, progress_message, status_iterator
from sphinx.util.console import darkgreen  # type: ignore
from sphinx.util.docutils import new_document
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.nodes import inline_all_toctrees
from sphinx.util.osutil import SEP, ensuredir, make_filename_from_project
from sphinx.writers.texinfo import TexinfoTranslator, TexinfoWriter
 Better default texinfo_documents settings. 