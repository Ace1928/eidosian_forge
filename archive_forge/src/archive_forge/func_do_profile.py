import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def do_profile():
    lsprof.profile(profiled)
    calls.append('after_profiled')