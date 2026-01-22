from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_permissionsFromStat(self) -> None:
    """
        L{Permissions}'s constructor takes a valid permissions bitmask and
        parsaes it to produce the correct set of boolean permissions.
        """

    def _rwxFromStat(statModeInt: int, who: str) -> filepath.RWX:

        def getPermissionBit(what: str, who: str) -> bool:
            constant: int = getattr(stat, f'S_I{what}{who}')
            return statModeInt & constant > 0
        return filepath.RWX(*(getPermissionBit(what, who) for what in ('R', 'W', 'X')))
    for u in range(0, 8):
        for g in range(0, 8):
            for o in range(0, 8):
                chmodString = '%d%d%d' % (u, g, o)
                chmodVal = int(chmodString, 8)
                perm = filepath.Permissions(chmodVal)
                self.assertEqual(perm.user, _rwxFromStat(chmodVal, 'USR'), f'{chmodString}: got user: {perm.user}')
                self.assertEqual(perm.group, _rwxFromStat(chmodVal, 'GRP'), f'{chmodString}: got group: {perm.group}')
                self.assertEqual(perm.other, _rwxFromStat(chmodVal, 'OTH'), f'{chmodString}: got other: {perm.other}')
    perm = filepath.Permissions(511)
    for who in ('user', 'group', 'other'):
        for what in ('read', 'write', 'execute'):
            self.assertTrue(getattr(getattr(perm, who), what))