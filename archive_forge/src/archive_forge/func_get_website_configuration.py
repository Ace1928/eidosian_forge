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
def get_website_configuration(self, headers=None):
    """Returns the current status of website configuration on the bucket.

        :param dict headers: Additional headers to send with the request.

        :rtype: dict
        :returns: A dictionary containing the parsed XML response from GCS. The
            overall structure is:

            * WebsiteConfiguration

              * MainPageSuffix: suffix that is appended to request that
                is for a "directory" on the website endpoint.
              * NotFoundPage: name of an object to serve when site visitors
                encounter a 404.
        """
    return self.get_website_configuration_with_xml(headers)[0]