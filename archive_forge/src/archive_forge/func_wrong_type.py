from __future__ import absolute_import
import types
from . import Errors
def wrong_type(self, num, value, expected):
    if type(value) == types.InstanceType:
        got = '%s.%s instance' % (value.__class__.__module__, value.__class__.__name__)
    else:
        got = type(value).__name__
    raise Errors.PlexTypeError('Invalid type for argument %d of Plex.%s (expected %s, got %s' % (num, self.__class__.__name__, expected, got))