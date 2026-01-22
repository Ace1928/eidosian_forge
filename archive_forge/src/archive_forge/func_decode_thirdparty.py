from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def decode_thirdparty(self, obj):
    if b'__thirdparty__' in obj:
        return ThirdParty(foo=obj[b'foo'])
    return obj