import sys
from oslo_serialization import jsonutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import common as exceptions
from oslo_messaging.tests import utils as test_utils
def test_deserialize_remote_exception(self):
    failure = {'class': self.clsname, 'module': self.modname, 'message': 'test', 'tb': ['traceback\ntraceback\n'], 'args': self.args, 'kwargs': self.kwargs}
    serialized = jsonutils.dumps(failure)
    ex = exceptions.deserialize_remote_exception(serialized, self.allowed)
    self.assertIsInstance(ex, self.cls)
    self.assertEqual(self.remote_name, ex.__class__.__name__)
    self.assertEqual(self.str, str(ex))
    if hasattr(self, 'msg'):
        self.assertEqual(self.msg, str(ex))
        self.assertEqual((self.msg,) + self.remote_args, ex.args)
    else:
        self.assertEqual(self.remote_args, ex.args)