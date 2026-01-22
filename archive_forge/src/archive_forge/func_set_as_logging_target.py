from __future__ import division
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import BotoClientError
from boto.s3.acl import Policy, CannedACLStrings, Grant
from boto.s3.key import Key
from boto.s3.prefix import Prefix
from boto.s3.deletemarker import DeleteMarker
from boto.s3.multipart import MultiPartUpload
from boto.s3.multipart import CompleteMultiPartUpload
from boto.s3.multidelete import MultiDeleteResult
from boto.s3.multidelete import Error
from boto.s3.bucketlistresultset import BucketListResultSet
from boto.s3.bucketlistresultset import VersionedBucketListResultSet
from boto.s3.bucketlistresultset import MultiPartUploadListResultSet
from boto.s3.lifecycle import Lifecycle
from boto.s3.tagging import Tags
from boto.s3.cors import CORSConfiguration
from boto.s3.bucketlogging import BucketLogging
from boto.s3 import website
import boto.jsonresponse
import boto.utils
import xml.sax
import xml.sax.saxutils
import re
import base64
from collections import defaultdict
from boto.compat import BytesIO, six, StringIO, urllib
from boto.utils import get_utf8able_str
def set_as_logging_target(self, headers=None):
    """
        Setup the current bucket as a logging target by granting the necessary
        permissions to the LogDelivery group to write log files to this bucket.
        """
    policy = self.get_acl(headers=headers)
    g1 = Grant(permission='WRITE', type='Group', uri=self.LoggingGroup)
    g2 = Grant(permission='READ_ACP', type='Group', uri=self.LoggingGroup)
    policy.acl.add_grant(g1)
    policy.acl.add_grant(g2)
    self.set_acl(policy, headers=headers)