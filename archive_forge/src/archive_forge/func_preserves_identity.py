from toolz import *
import toolz
import toolz.curried
import pickle
from toolz.utils import raises
def preserves_identity(obj):
    return pickle.loads(pickle.dumps(obj)) is obj