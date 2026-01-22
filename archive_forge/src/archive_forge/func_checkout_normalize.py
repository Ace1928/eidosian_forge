from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
def checkout_normalize(self, blob, tree_path):
    """Normalize a blob during a checkout operation."""
    if self.fallback_read_filter is not None:
        return normalize_blob(blob, self.fallback_read_filter, binary_detection=True)
    return blob