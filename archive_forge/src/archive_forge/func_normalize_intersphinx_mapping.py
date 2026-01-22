import concurrent.futures
import functools
import posixpath
import re
import sys
import time
from os import path
from types import ModuleType
from typing import IO, Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlsplit, urlunsplit
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.utils import Reporter, relative_path
import sphinx
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders.html import INVENTORY_FILENAME
from sphinx.config import Config
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError
from sphinx.locale import _, __
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging, requests
from sphinx.util.docutils import CustomReSTDispatcher, SphinxRole
from sphinx.util.inventory import InventoryFile
from sphinx.util.typing import Inventory, InventoryItem, RoleFunction
def normalize_intersphinx_mapping(app: Sphinx, config: Config) -> None:
    for key, value in config.intersphinx_mapping.copy().items():
        try:
            if isinstance(value, (list, tuple)):
                name, (uri, inv) = (key, value)
                if not isinstance(name, str):
                    logger.warning(__('intersphinx identifier %r is not string. Ignored'), name)
                    config.intersphinx_mapping.pop(key)
                    continue
            else:
                name, uri, inv = (None, key, value)
            if not isinstance(inv, tuple):
                config.intersphinx_mapping[key] = (name, (uri, (inv,)))
            else:
                config.intersphinx_mapping[key] = (name, (uri, inv))
        except Exception as exc:
            logger.warning(__('Failed to read intersphinx_mapping[%s], ignored: %r'), key, exc)
            config.intersphinx_mapping.pop(key)