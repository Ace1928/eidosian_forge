import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def show_string(self, delta, *args, **kwargs):
    to_file = StringIO()
    _mod_delta.report_delta(to_file, delta, *args, **kwargs)
    return to_file.getvalue()