from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def invAddEntry(self, inv, path, file_id=None):
    if path.endswith('/'):
        path = path[:-1]
        kind = 'directory'
    else:
        kind = 'file'
    parent_path, basename = osutils.split(path)
    parent_id = inv.path2id(parent_path)
    inv.add(inventory.make_entry(kind, basename, parent_id, file_id))