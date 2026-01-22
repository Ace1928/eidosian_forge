import itertools
from testtools import matchers
from pbr.tests import base
from pbr import version
from_pip_string = version.SemanticVersion.from_pip_string
def test_from_pip_string_pure_git_hash(self):
    self.assertRaises(ValueError, from_pip_string, '6eed5ae')