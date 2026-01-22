import tarfile
import zipfile
from .. import export, filter_tree, tests
from . import fixtures
from .test_filters import _stack_1
def stack_callback(path):
    return _stack_1