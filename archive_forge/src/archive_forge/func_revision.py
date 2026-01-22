from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def revision(self):
    """Revision this variation of the file was introduced in."""
    if self._revision is None:
        rev_id = self.revision_id()
        if rev_id is not None:
            repo = getattr(self._tree, '_repository', None)
            if repo is None:
                repo = self._tree.branch.repository
            self._revision = repo.get_revision(rev_id)
    return self._revision