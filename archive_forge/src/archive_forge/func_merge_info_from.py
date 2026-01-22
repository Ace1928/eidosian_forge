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
def merge_info_from(self, docnames: List[str], other: 'BuildEnvironment', app: 'Sphinx') -> None:
    """Merge global information gathered about *docnames* while reading them
        from the *other* environment.

        This possibly comes from a parallel build process.
        """
    docnames = set(docnames)
    for docname in docnames:
        self.all_docs[docname] = other.all_docs[docname]
        self.included[docname] = other.included[docname]
        if docname in other.reread_always:
            self.reread_always.add(docname)
    for domainname, domain in self.domains.items():
        domain.merge_domaindata(docnames, other.domaindata[domainname])
    self.events.emit('env-merge-info', self, docnames, other)