from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_auth_private_unowned(self):
    """
        Tests that an authenticated context (with is_admin set to
        False) cannot access an image (which it does not own) with
        is_public set to False.
        """
    self.do_visible(False, 'pattieblack', False, project_id='froggy')