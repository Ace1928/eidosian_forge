import uuid
from oslo_utils import timeutils
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib import trusts
def test_list_filter_trustor(self):
    expected_query = {'trustor_user_id': '12345'}
    super(TrustTests, self).test_list(expected_query=expected_query, trustor_user='12345')