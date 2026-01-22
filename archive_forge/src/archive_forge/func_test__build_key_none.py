import os
from unittest import mock
import dogpile.cache
from ironicclient.common import filecache
from ironicclient.tests.unit import utils
def test__build_key_none(self):
    result = filecache._build_key(None, None)
    self.assertEqual('None:None', result)