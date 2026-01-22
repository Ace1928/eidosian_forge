import time
from unittest import mock
import uuid
from pycadf import attachment
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import event
from pycadf import geolocation
from pycadf import host
from pycadf import identifier
from pycadf import measurement
from pycadf import metric
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf.tests import base
from pycadf import timestamp
@mock.patch('pycadf.identifier.warnings.warn')
def test_identifier_long_nonjoined_uuid_is_invalid(self, warning_mock):
    char_42_id = '3adce28e67e44544a5a9d5f1ab54f578a86d310aac'
    self.assertTrue(identifier.is_valid(char_42_id))
    self.assertTrue(warning_mock.called)