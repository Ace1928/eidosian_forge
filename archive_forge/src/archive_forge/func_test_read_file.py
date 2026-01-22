import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_read_file(self):
    f = BytesIO(b'\n# a comment\n  \n# and an empty line:\n\n\\#not a comment\n!negative\nwith trailing whitespace \nwith escaped trailing whitespace\\ \n')
    self.assertEqual(list(read_ignore_patterns(f)), [b'\\#not a comment', b'!negative', b'with trailing whitespace', b'with escaped trailing whitespace '])