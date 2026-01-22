from taskflow import test
from taskflow.utils import misc
def test_unicode_binary(self):
    self._check(_bytes('привет'), u'привет')