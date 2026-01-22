import codecs
import pickle
import time
import warnings
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import DependencyList
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import CONFIG_CHANGED_REASON, CONFIG_OK, BuildEnvironment
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import SphinxError
from sphinx.events import EventManager
from sphinx.locale import __
from sphinx.util import (UnicodeDecodeErrorHandler, get_filetype, import_object, logging,
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import sphinx_domains
from sphinx.util.i18n import CatalogInfo, CatalogRepository, docname_to_domain
from sphinx.util.osutil import SEP, ensuredir, relative_uri, relpath
from sphinx.util.parallel import ParallelTasks, SerialTasks, make_chunks, parallel_available
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType
from sphinx import directives  # NOQA isort:skip
from sphinx import roles  # NOQA isort:skip
def build_specific(self, filenames: List[str]) -> None:
    """Only rebuild as much as needed for changes in the *filenames*."""
    docnames: List[str] = []
    for filename in filenames:
        filename = path.normpath(path.abspath(filename))
        if not path.isfile(filename):
            logger.warning(__('file %r given on command line does not exist, '), filename)
            continue
        if not filename.startswith(self.srcdir):
            logger.warning(__('file %r given on command line is not under the source directory, ignoring'), filename)
            continue
        docname = self.env.path2doc(filename)
        if not docname:
            logger.warning(__('file %r given on command line is not a valid document, ignoring'), filename)
            continue
        docnames.append(docname)
    self.compile_specific_catalogs(filenames)
    self.build(docnames, method='specific', summary=__('%d source files given on command line') % len(docnames))