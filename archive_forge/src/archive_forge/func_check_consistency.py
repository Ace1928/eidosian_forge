import os
import pickle
import warnings
from collections import defaultdict
from copy import copy
from datetime import datetime
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, Optional,
import docutils
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import BuildEnvironmentError, DocumentError, ExtensionError, SphinxError
from sphinx.events import EventManager
from sphinx.locale import __
from sphinx.project import Project
from sphinx.transforms import SphinxTransformer
from sphinx.util import DownloadFiles, FilenameUniqDict, logging
from sphinx.util.docutils import LoggingReporter
from sphinx.util.i18n import CatalogRepository, docname_to_domain
from sphinx.util.nodes import is_translatable
from sphinx.util.osutil import canon_path, os_path
def check_consistency(self) -> None:
    """Do consistency checks."""
    included = set().union(*self.included.values())
    for docname in sorted(self.all_docs):
        if docname not in self.files_to_rebuild:
            if docname == self.config.root_doc:
                continue
            if docname in included:
                continue
            if 'orphan' in self.metadata[docname]:
                continue
            logger.warning(__("document isn't included in any toctree"), location=docname)
    for domain in self.domains.values():
        domain.check_consistency()
    self.events.emit('env-check-consistency', self)