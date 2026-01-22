from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import urllib.parse
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import requests
def get_signing_information_from_json(raw_data, password_bytes=None):
    """Loads signing information from a JSON or P12 private key.

  JSON keys from GCP do not use a passphrase by default, so we follow gsutil in
  not prompting the user for a password.

  P12 keystores from GCP do use a default ('notasecret'), so we will prompt the
  user if they do not provide a password.

  Args:
    raw_data (str): Un-parsed JSON data from the key file or creds store.
    password_bytes (bytes): A password used to decrypt encrypted private keys.

  Returns:
    A tuple (client_id: str, key: crypto.PKey), which can be used to sign URLs.
  """
    from OpenSSL import crypto
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID
    try:
        parsed_json = json.loads(raw_data)
        client_id = parsed_json[JSON_CLIENT_ID_KEY]
        key = crypto.load_privatekey(crypto.FILETYPE_PEM, parsed_json[JSON_PRIVATE_KEY_KEY], passphrase=password_bytes)
        return (client_id, key)
    except ValueError:
        if not password_bytes:
            password_bytes = console_io.PromptPassword("Keystore password (default: 'notasecret'): ")
        if not isinstance(password_bytes, bytes):
            password_bytes = password_bytes.encode('utf-8')
        private_key, certificate, _ = pkcs12.load_key_and_certificates(raw_data, password=password_bytes)
        client_id = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        return (client_id[0].value, private_key)