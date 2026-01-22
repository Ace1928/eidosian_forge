import io
from oslotest import base
from oslo_privsep import comm
def test_badobj(self):

    class UnknownClass(object):
        pass
    obj = UnknownClass()
    self.assertRaises(TypeError, self.send, obj)