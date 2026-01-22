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
def set_xml_tags(self, tag_str, headers=None, query_args='tagging'):
    if headers is None:
        headers = {}
    md5 = boto.utils.compute_md5(StringIO(tag_str))
    headers['Content-MD5'] = md5[1]
    headers['Content-Type'] = 'text/xml'
    if not isinstance(tag_str, bytes):
        tag_str = tag_str.encode('utf-8')
    response = self.connection.make_request('PUT', self.name, data=tag_str, query_args=query_args, headers=headers)
    body = response.read()
    if response.status != 204:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)
    return True