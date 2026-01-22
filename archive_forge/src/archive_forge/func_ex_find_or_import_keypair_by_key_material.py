import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def ex_find_or_import_keypair_by_key_material(self, pubkey):
    """
        Given a public key, look it up in the EC2 KeyPair database. If it
        exists, return any information we have about it. Otherwise, create it.

        Keys that are created are named based on their comment and fingerprint.

        :rtype: ``dict``
        """
    key_fingerprint = get_pubkey_ssh2_fingerprint(pubkey)
    key_comment = get_pubkey_comment(pubkey, default='unnamed')
    key_name = '{}-{}'.format(key_comment, key_fingerprint)
    key_pairs = self.list_key_pairs()
    key_pairs = [key_pair for key_pair in key_pairs if key_pair.fingerprint == key_fingerprint]
    if len(key_pairs) >= 1:
        key_pair = key_pairs[0]
        result = {'keyName': key_pair.name, 'keyFingerprint': key_pair.fingerprint}
    else:
        result = self.ex_import_keypair_from_string(key_name, pubkey)
    return result