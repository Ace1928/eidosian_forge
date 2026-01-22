from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
def get_checkin_filter_autocrlf(core_autocrlf):
    """Returns the correct checkin filter base on autocrlf value.

    Args:
      core_autocrlf: The bytes configuration value of core.autocrlf.
        Valid values are: b'true', b'false' or b'input'.
    Returns: Either None if no filter has to be applied or a function
        accepting a single argument, a binary text hunk
    """
    if core_autocrlf == b'true' or core_autocrlf == b'input':
        return convert_crlf_to_lf
    return None