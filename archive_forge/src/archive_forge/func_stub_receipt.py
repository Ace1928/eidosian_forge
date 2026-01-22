import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def stub_receipt(self, receipt=None, receipt_data=None, **kwargs):
    if not receipt:
        receipt = self.TEST_RECEIPT
    if not receipt_data:
        receipt_data = self.TEST_RECEIPT_RESPONSE
    self.stub_url('POST', ['auth', 'tokens'], headers={'Openstack-Auth-Receipt': receipt}, status_code=401, json=receipt_data, **kwargs)