from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _lazy
from oslo_i18n import _message
def test_change_lazy(self):
    _lazy.enable_lazy(True)
    tf = _factory.TranslatorFactory('domain')
    r = tf.primary('some text')
    self.assertIsInstance(r, _message.Message)
    _lazy.enable_lazy(False)
    r = tf.primary('some text')
    self.assertNotIsInstance(r, _message.Message)