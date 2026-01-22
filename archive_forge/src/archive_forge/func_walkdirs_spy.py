import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def walkdirs_spy(*args, **kwargs):
    for val in orig(*args, **kwargs):
        returned.append(val[0][0])
        yield val