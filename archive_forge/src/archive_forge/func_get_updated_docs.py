from typing import TYPE_CHECKING, Dict, List, Optional, Set
from docutils import nodes
from sphinx.environment import BuildEnvironment
def get_updated_docs(self, app: 'Sphinx', env: BuildEnvironment) -> List[str]:
    """Return a list of docnames to re-read.

        This methods is called after reading the whole of documents (experimental).
        """
    return []