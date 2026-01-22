import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_regex_special(self):
    filter = IgnoreFilter([b'/foo\\[bar\\]', b'/foo'])
    self.assertTrue(filter.is_ignored('foo'))
    self.assertTrue(filter.is_ignored('foo[bar]'))