from toolz import *
import toolz
import toolz.curried
import pickle
from toolz.utils import raises
def test_curried_exceptions():
    merge = pickle.loads(pickle.dumps(toolz.curried.merge))
    assert merge is toolz.curried.merge