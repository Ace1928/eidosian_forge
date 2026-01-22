import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def locked_message(a_bool):
    if a_bool:
        return 'locked'
    else:
        return 'unlocked'