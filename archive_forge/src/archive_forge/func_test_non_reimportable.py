import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
def test_non_reimportable(self):

    def factory():
        pass
    self.assertRaisesRegex(ValueError, 'Flow factory .* is not reimportable', taskflow.engines.load_from_factory, factory)