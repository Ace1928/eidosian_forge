from ..bisect_multi import bisect_multi_bytes
from . import TestCase
Doing a lookup in a zero-length file still does a single request.

        This makes sense because the bisector cannot tell how long content is
        and its more flexible to only stop when the content object says 'False'
        for a given location, key pair.
        