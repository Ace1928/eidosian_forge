from cryptography import exceptions as crypto_exception
import glance_store as store
from unittest import mock
import urllib
from oslo_config import cfg
from oslo_policy import policy
from glance.async_.flows._internal_plugins import base_download
from glance.common import exception
from glance.common import store_utils
from glance.common import wsgi
import glance.context
import glance.db.simple.api as simple_db
def fake_get_verifier(context, img_signature_certificate_uuid, img_signature_hash_method, img_signature, img_signature_key_type):
    verifier = mock.Mock()
    if img_signature is not None and img_signature == 'VALID':
        verifier.verify.return_value = None
    else:
        ex = crypto_exception.InvalidSignature()
        verifier.verify.side_effect = ex
    return verifier