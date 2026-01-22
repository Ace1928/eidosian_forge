from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def make_branch(self, path, format='git'):
    return super().make_branch(path, format=format)