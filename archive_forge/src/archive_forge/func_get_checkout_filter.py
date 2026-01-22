from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
def get_checkout_filter(core_eol, core_autocrlf, git_attributes):
    """Returns the correct checkout filter based on the passed arguments."""
    return get_checkout_filter_autocrlf(core_autocrlf)