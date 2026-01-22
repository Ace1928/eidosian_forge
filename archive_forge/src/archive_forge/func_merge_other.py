from typing import TYPE_CHECKING, Dict, List, Optional, Set
from docutils import nodes
from sphinx.environment import BuildEnvironment
def merge_other(self, app: 'Sphinx', env: BuildEnvironment, docnames: Set[str], other: BuildEnvironment) -> None:
    """Merge in specified data regarding docnames from a different `BuildEnvironment`
        object which coming from a subprocess in parallel builds."""
    raise NotImplementedError