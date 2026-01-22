import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_failsWithDifferentModule(self):
    """
        L{isOriginalLocation} returns False when the attribute refers to
        an object outside of the module where that object was defined.
        """
    originalSource = '        class ImportThisClass(object):\n            pass\n        importThisObject = ImportThisClass()\n        importThisNestingObject = ImportThisClass()\n        importThisNestingObject.nestedObject = ImportThisClass()\n        '
    importingSource = '        from original import (ImportThisClass,\n                              importThisObject,\n                              importThisNestingObject)\n        '
    self.makeModule(originalSource, self.pathDir, 'original.py')
    importingDict = self.makeModuleAsDict(importingSource, self.pathDir, 'importing.py')
    self.assertFalse(self.isOriginalLocation(importingDict['importing.ImportThisClass']))
    self.assertFalse(self.isOriginalLocation(importingDict['importing.importThisObject']))
    nestingObject = importingDict['importing.importThisNestingObject']
    nestingObjectDict = self.attributesAsDict(nestingObject)
    nestedObject = nestingObjectDict['importing.importThisNestingObject.nestedObject']
    self.assertFalse(self.isOriginalLocation(nestedObject))