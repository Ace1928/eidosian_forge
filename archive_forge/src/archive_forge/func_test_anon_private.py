from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_anon_private(self):
    """
        Tests that an anonymous context (with is_admin set to False)
        can access an unowned image with is_public set to False.
        """
    self.do_visible(True, None, False)