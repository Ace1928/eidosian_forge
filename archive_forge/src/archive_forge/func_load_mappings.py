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
def load_mappings(app: Sphinx) -> None:
    """Load all intersphinx mappings into the environment."""
    now = int(time.time())
    inventories = InventoryAdapter(app.builder.env)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = []
        for name, (uri, invs) in app.config.intersphinx_mapping.values():
            futures.append(pool.submit(fetch_inventory_group, name, uri, invs, inventories.cache, app, now))
        updated = [f.result() for f in concurrent.futures.as_completed(futures)]
    if any(updated):
        inventories.clear()
        cached_vals = list(inventories.cache.values())
        named_vals = sorted((v for v in cached_vals if v[0]))
        unnamed_vals = [v for v in cached_vals if not v[0]]
        for name, _x, invdata in named_vals + unnamed_vals:
            if name:
                inventories.named_inventory[name] = invdata
            for type, objects in invdata.items():
                inventories.main_inventory.setdefault(type, {}).update(objects)