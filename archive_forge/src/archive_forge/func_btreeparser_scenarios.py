import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def btreeparser_scenarios():
    import breezy.bzr._btree_serializer_py as py_module
    scenarios = [('python', {'parse_btree': py_module})]
    if compiled_btreeparser_feature.available():
        scenarios.append(('C', {'parse_btree': compiled_btreeparser_feature.module}))
    return scenarios