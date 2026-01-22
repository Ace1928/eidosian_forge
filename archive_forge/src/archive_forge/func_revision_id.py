from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def revision_id(self):
    """Id of revision that last changed this file."""
    if self._revision_id is None:
        if self._tree is not None:
            self._revision_id = self._tree.get_file_revision(self._relpath)
    return self._revision_id