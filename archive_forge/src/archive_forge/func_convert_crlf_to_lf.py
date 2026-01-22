from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
def convert_crlf_to_lf(text_hunk):
    """Convert CRLF in text hunk into LF.

    Args:
      text_hunk: A bytes string representing a text hunk
    Returns: The text hunk with the same type, with CRLF replaced into LF
    """
    return text_hunk.replace(CRLF, LF)