import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
def test_empty_text_completedefault(self):
    self._test_completedefault(['file', 'folder', ' long'], 'show ', 5)