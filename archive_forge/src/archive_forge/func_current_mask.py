from traitlets import Bool, Set
from .base import Preprocessor
def current_mask(self, mask):
    """Get the current mask for a mask."""
    return {self.current_key(k) for k in mask if self.current_key(k) is not None}