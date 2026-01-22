import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
def parse_NoneTrueFalse(self, arg):
    if not arg:
        return None
    if arg == b'False':
        return False
    if arg == b'True':
        return True
    raise AssertionError('invalid arg %r' % arg)