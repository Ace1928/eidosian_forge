from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def get_file_lines(self, revision_id, path):
    """Get the lines stored for a file in a given revision."""
    revtree = self.repo.revision_tree(revision_id)
    return revtree.get_file_lines(path)