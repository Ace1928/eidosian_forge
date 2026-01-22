import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def testMalformedSubmodules(self):
    cf = ConfigFile.from_file(BytesIO(b'[submodule "core/lib"]\n\tpath = core/lib\n\turl = https://github.com/phhusson/QuasselC.git\n\n[submodule "dulwich"]\n\turl = https://github.com/jelmer/dulwich\n'))
    got = list(parse_submodules(cf))
    self.assertEqual([(b'core/lib', b'https://github.com/phhusson/QuasselC.git', b'core/lib')], got)