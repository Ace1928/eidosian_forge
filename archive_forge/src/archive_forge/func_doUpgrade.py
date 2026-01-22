import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def doUpgrade():
    global versionedsToUpgrade, upgraded
    for versioned in list(versionedsToUpgrade.values()):
        requireUpgrade(versioned)
    versionedsToUpgrade = {}
    upgraded = {}