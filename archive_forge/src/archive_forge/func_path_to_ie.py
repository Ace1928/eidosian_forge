import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def path_to_ie(self, path, file_id, rev_id, dir_ids):
    if path.endswith('/'):
        is_dir = True
        path = path[:-1]
    else:
        is_dir = False
    dirname, basename = osutils.split(path)
    try:
        dir_id = dir_ids[dirname]
    except KeyError:
        dir_id = osutils.basename(dirname).encode('utf-8') + b'-id'
    if is_dir:
        ie = inventory.InventoryDirectory(file_id, basename, dir_id)
        dir_ids[path] = file_id
    else:
        ie = inventory.InventoryFile(file_id, basename, dir_id)
        ie.text_size = 0
        ie.text_sha1 = b''
    ie.revision = rev_id
    return ie