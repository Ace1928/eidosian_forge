from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def source_tree(self):
    """Source Tree object."""
    return self._tree