from traitlets import Bool, Set
from .base import Preprocessor
def nested_masks(self, mask):
    """Get the nested masks for a mask."""
    return {self.current_key(k[0]): k[1:] for k in mask if k and (not isinstance(k, str)) and (len(k) > 1)}