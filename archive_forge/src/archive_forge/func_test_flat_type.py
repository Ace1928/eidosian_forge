import re
import unittest
from wsme import exc
from wsme import types
def test_flat_type(self):

    class Flat(object):
        aint = int
        abytes = bytes
        atext = str
        afloat = float
    types.register_type(Flat)
    assert len(Flat._wsme_attributes) == 4
    attrs = Flat._wsme_attributes
    print(attrs)
    assert attrs[0].key == 'aint'
    assert attrs[0].name == 'aint'
    assert isinstance(attrs[0], types.wsattr)
    assert attrs[0].datatype == int
    assert attrs[0].mandatory is False
    assert attrs[1].key == 'abytes'
    assert attrs[1].name == 'abytes'
    assert attrs[2].key == 'atext'
    assert attrs[2].name == 'atext'
    assert attrs[3].key == 'afloat'
    assert attrs[3].name == 'afloat'