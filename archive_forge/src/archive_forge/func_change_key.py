from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
def change_key(change):
    if change.path[0] is None:
        path = change.path[1]
    else:
        path = change.path[0]
    return (path, change.file_id)