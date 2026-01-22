import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def test_report_changes(self):
    """Test change detection of report_changes"""
    self.assertChangesEqual(modified='unchanged', renamed=False, versioned_change='unchanged', exe_change=False)
    self.assertChangesEqual(modified='kind changed', kind=('file', 'directory'))
    self.assertChangesEqual(modified='created', kind=(None, 'directory'))
    self.assertChangesEqual(modified='deleted', kind=('directory', None))
    self.assertChangesEqual(content_change=True, modified='modified')
    self.assertChangesEqual(renamed=True, name=('old', 'new'))
    self.assertChangesEqual(renamed=True, parent_id=('old-parent', 'new-parent'))
    self.assertChangesEqual(versioned_change='added', versioned=(False, True))
    self.assertChangesEqual(versioned_change='removed', versioned=(True, False))
    self.assertChangesEqual(exe_change=True, executable=(True, False))
    self.assertChangesEqual(exe_change=False, executable=(True, False), kind=('directory', 'directory'))
    self.assertChangesEqual(exe_change=False, modified='kind changed', executable=(False, True), kind=('directory', 'file'))
    self.assertChangesEqual(parent_id=('pid', None))
    self.assertChangesEqual(versioned_change='removed', modified='deleted', versioned=(True, False), kind=('directory', None))
    self.assertChangesEqual(versioned_change='removed', modified='created', versioned=(True, False), kind=(None, 'file'))
    self.assertChangesEqual(versioned_change='removed', modified='modified', renamed=True, exe_change=True, versioned=(True, False), content_change=True, name=('old', 'new'), executable=(False, True))