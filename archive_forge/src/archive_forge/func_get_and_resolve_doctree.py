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
def get_and_resolve_doctree(self, docname: str, builder: 'Builder', doctree: Optional[nodes.document]=None, prune_toctrees: bool=True, includehidden: bool=False) -> nodes.document:
    """Read the doctree from the pickle, resolve cross-references and
        toctrees and return it.
        """
    if doctree is None:
        doctree = self.get_doctree(docname)
    self.apply_post_transforms(doctree, docname)
    for toctreenode in doctree.findall(addnodes.toctree):
        result = TocTree(self).resolve(docname, builder, toctreenode, prune=prune_toctrees, includehidden=includehidden)
        if result is None:
            toctreenode.replace_self([])
        else:
            toctreenode.replace_self(result)
    return doctree