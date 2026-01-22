import os
import stat
from . import errors, lock
from . import revision as _mod_revision
from . import transport as _mod_transport
from .bzr.inventory import Inventory
from .bzr.inventorytree import MutableInventoryTree
from .osutils import sha_file
from .transport.memory import MemoryTransport
@staticmethod
def create_on_branch(branch):
    """Create a MemoryTree for branch, using the last-revision of branch."""
    return MemoryTree(branch, branch.last_revision())