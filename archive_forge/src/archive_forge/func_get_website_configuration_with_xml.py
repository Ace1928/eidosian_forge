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
def get_website_configuration_with_xml(self, headers=None):
    """Returns the current status of website configuration on the bucket as
        unparsed XML.

        :param dict headers: Additional headers to send with the request.

        :rtype: 2-Tuple
        :returns: 2-tuple containing:

            1) A dictionary containing the parsed XML response from GCS. The
              overall structure is:

              * WebsiteConfiguration

                * MainPageSuffix: suffix that is appended to request that is for
                  a "directory" on the website endpoint.
                * NotFoundPage: name of an object to serve when site visitors
                  encounter a 404

            2) Unparsed XML describing the bucket's website configuration.
        """
    response = self.connection.make_request('GET', self.name, query_args='websiteConfig', headers=headers)
    body = response.read()
    boto.log.debug(body)
    if response.status != 200:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)
    e = boto.jsonresponse.Element()
    h = boto.jsonresponse.XmlHandler(e, None)
    h.parse(body)
    return (e, body)