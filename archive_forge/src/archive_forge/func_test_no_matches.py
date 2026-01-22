import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_no_matches(self):
    for path, pattern in NEGATIVE_MATCH_TESTS:
        self.assertFalse(match_pattern(path, pattern), f'path: {path!r}, pattern: {pattern!r}')