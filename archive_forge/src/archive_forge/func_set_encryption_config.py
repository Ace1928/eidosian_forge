from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import xml.sax
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import GSResponseError
from boto.exception import InvalidAclError
from boto.gs.acl import ACL, CannedACLStrings
from boto.gs.acl import SupportedPermissions as GSPermissions
from boto.gs.bucketlistresultset import VersionedBucketListResultSet
from boto.gs.cors import Cors
from boto.gs.encryptionconfig import EncryptionConfig
from boto.gs.lifecycle import LifecycleConfig
from boto.gs.key import Key as GSKey
from boto.s3.acl import Policy
from boto.s3.bucket import Bucket as S3Bucket
from boto.utils import get_utf8able_str
from boto.compat import quote
from boto.compat import six
def set_encryption_config(self, default_kms_key_name=None, headers=None):
    """Sets a bucket's EncryptionConfig XML document.

        :param str default_kms_key_name: A string containing a fully-qualified
            Cloud KMS key name.
        :param dict headers: Additional headers to send with the request.
        """
    body = self._construct_encryption_config_xml(default_kms_key_name=default_kms_key_name)
    response = self.connection.make_request('PUT', self.name, data=body, query_args=ENCRYPTION_CONFIG_ARG, headers=headers)
    body = response.read()
    if response.status != 200:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)