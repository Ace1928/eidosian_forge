from unittest import mock
from osprofiler.drivers.mongodb import MongoDB
from osprofiler.tests import test
def test_build_empty_tree(self):
    self.assertEqual([], self.mongodb._build_tree({}))