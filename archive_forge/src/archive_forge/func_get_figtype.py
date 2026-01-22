from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.environment.adapters.toctree import TocTree
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.locale import __
from sphinx.transforms import SphinxContentsFilter
from sphinx.util import logging, url_re
def get_figtype(node: Node) -> Optional[str]:
    for domain in env.domains.values():
        figtype = domain.get_enumerable_node_type(node)
        if domain.name == 'std' and (not domain.get_numfig_title(node)):
            continue
        if figtype:
            return figtype
    return None