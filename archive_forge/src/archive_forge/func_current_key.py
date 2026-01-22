from traitlets import Bool, Set
from .base import Preprocessor
def current_key(self, mask_key):
    """Get the current key for a mask key."""
    if isinstance(mask_key, str):
        return mask_key
    if len(mask_key) == 0:
        return None
    return mask_key[0]