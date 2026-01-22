import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
Adding unnormalized unicode filenames fail if and only if the
        workingtree format has the requires_normalized_unicode_filenames flag
        set and the underlying filesystem doesn't normalize.
        