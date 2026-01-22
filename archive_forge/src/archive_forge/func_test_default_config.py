import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_default_config(self):
    cf = self.from_file(b'[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n\tlogallrefupdates = true\n')
    self.assertEqual(ConfigFile({(b'core',): {b'repositoryformatversion': b'0', b'filemode': b'true', b'bare': b'false', b'logallrefupdates': b'true'}}), cf)