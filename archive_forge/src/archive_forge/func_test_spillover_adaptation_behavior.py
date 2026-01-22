import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_spillover_adaptation_behavior(self):
    ex = self.examples
    self.adaptation_manager.register_factory(factory=ex.FileTypeToIEditor, from_protocol=ex.FileType, to_protocol=ex.IEditor)
    self.adaptation_manager.register_factory(factory=ex.IScriptableToIUndoable, from_protocol=ex.IScriptable, to_protocol=ex.IUndoable)
    file_type = ex.FileType()
    printable = self.adaptation_manager.adapt(file_type, ex.IUndoable, None)
    self.assertIsNone(printable)