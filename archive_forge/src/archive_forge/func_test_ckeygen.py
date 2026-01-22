from unittest import skipIf
from twisted.python.reflect import requireModule
from twisted.python.test.test_shellcomp import ZshScriptTestMixin
from twisted.scripts.test.test_scripts import ScriptTestsMixin
from twisted.trial.unittest import TestCase
def test_ckeygen(self) -> None:
    self.scriptTest('conch/ckeygen')