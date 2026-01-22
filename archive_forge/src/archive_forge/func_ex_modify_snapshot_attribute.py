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
def ex_modify_snapshot_attribute(self, snapshot, attributes):
    """
        Modify Snapshot attributes.

        :param      snapshot: VolumeSnapshot instance
        :type       snanpshot: :class:`VolumeSnapshot`

        :param      attributes: Dictionary with snapshot attributes
        :type       attributes: ``dict``

        :return: True on success, False otherwise.
        :rtype: ``bool``
        """
    attributes = attributes or {}
    attributes.update({'SnapshotId': snapshot.id})
    params = {'Action': 'ModifySnapshotAttribute'}
    params.update(attributes)
    res = self.connection.request(self.path, params=params.copy()).object
    return self._get_boolean(res)