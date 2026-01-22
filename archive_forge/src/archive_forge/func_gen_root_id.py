from ..lazy_import import lazy_import
import time
from breezy import (
from .. import lazy_regex
def gen_root_id():
    """Return a new tree-root file id."""
    return gen_file_id('tree_root')