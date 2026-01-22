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
def collect_relations(self) -> Dict[str, List[Optional[str]]]:
    traversed = set()

    def traverse_toctree(parent: Optional[str], docname: str) -> Iterator[Tuple[Optional[str], str]]:
        if parent == docname:
            logger.warning(__('self referenced toctree found. Ignored.'), location=docname, type='toc', subtype='circular')
            return
        yield (parent, docname)
        traversed.add(docname)
        for child in self.toctree_includes.get(docname) or []:
            for subparent, subdocname in traverse_toctree(docname, child):
                if subdocname not in traversed:
                    yield (subparent, subdocname)
                    traversed.add(subdocname)
    relations = {}
    docnames = traverse_toctree(None, self.config.root_doc)
    prevdoc = None
    parent, docname = next(docnames)
    for nextparent, nextdoc in docnames:
        relations[docname] = [parent, prevdoc, nextdoc]
        prevdoc = docname
        docname = nextdoc
        parent = nextparent
    relations[docname] = [parent, prevdoc, None]
    return relations