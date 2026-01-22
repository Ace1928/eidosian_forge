from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_auth_public(self):
    """
        Tests that an authenticated context (with is_admin set to
        False) can access an image with is_public set to True.
        """
    self.do_visible(True, None, True, project_id='froggy')