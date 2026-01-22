import sys
from types import ModuleType
from twisted.trial.unittest import TestCase
def test_packageWithModule(self):
    """
        Processing of the attributes dictionary is recursive, so a C{dict} value
        it contains may itself contain a C{dict} value to the same effect.
        """
    modules = {}
    _makePackages(None, dict(twisted=dict(web=dict(version='321'))), modules)
    self.assertIsInstance(modules, dict)
    self.assertIsInstance(modules['twisted'], ModuleType)
    self.assertEqual('twisted', modules['twisted'].__name__)
    self.assertIsInstance(modules['twisted'].web, ModuleType)
    self.assertEqual('twisted.web', modules['twisted'].web.__name__)
    self.assertEqual('321', modules['twisted'].web.version)