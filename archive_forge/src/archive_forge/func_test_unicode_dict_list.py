from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
def test_unicode_dict_list(self):
    inner = {'key1': 'somevalue', 'key2': 'somevalue'}
    m = {'topkey': inner, 'list': ['somevalue', '2'], 'u': '2'}
    glance_store.check_location_metadata(m)