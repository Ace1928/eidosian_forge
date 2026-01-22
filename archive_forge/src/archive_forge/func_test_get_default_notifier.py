from unittest import mock
from osprofiler import notifier
from osprofiler.tests import test
def test_get_default_notifier(self):
    self.assertEqual(notifier.get(), notifier._noop_notifier)