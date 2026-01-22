from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
def test_FileRebuild(self) -> None:
    import shutil
    import time
    from twisted.python.util import sibpath
    shutil.copyfile(sibpath(__file__, 'myrebuilder1.py'), os.path.join(self.fakelibPath, 'myrebuilder.py'))
    from twisted_rebuild_fakelib import myrebuilder
    a = myrebuilder.A()
    b = myrebuilder.B()
    i = myrebuilder.Inherit()
    self.assertEqual(a.a(), 'a')
    time.sleep(1.1)
    shutil.copyfile(sibpath(__file__, 'myrebuilder2.py'), os.path.join(self.fakelibPath, 'myrebuilder.py'))
    rebuild.rebuild(myrebuilder)
    b2 = myrebuilder.B()
    self.assertEqual(b2.b(), 'c')
    self.assertEqual(b.b(), 'c')
    self.assertEqual(i.a(), 'd')
    self.assertEqual(a.a(), 'b')