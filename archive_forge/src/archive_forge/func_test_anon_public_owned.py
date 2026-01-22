from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_anon_public_owned(self):
    """
        Tests that an anonymous context (with is_admin set to False)
        can access an owned image with is_public set to True.
        """
    self.do_visible(True, 'pattieblack', True)