import xml.sax
import base64
import time
from boto.compat import six, urllib
from boto.auth import detect_potential_s3sigv4
import boto.utils
from boto.connection import AWSAuthConnection
from boto import handler
from boto.s3.bucket import Bucket
from boto.s3.key import Key
from boto.resultset import ResultSet
from boto.exception import BotoClientError, S3ResponseError
from boto.utils import get_utf8able_str
def generate_url_sigv4(self, expires_in, method, bucket='', key='', headers=None, force_http=False, response_headers=None, version_id=None, iso_date=None):
    path = self.calling_format.build_path_base(bucket, key)
    auth_path = self.calling_format.build_auth_path(bucket, key)
    host = self.calling_format.build_host(self.server_name(), bucket)
    if host.endswith(':443'):
        host = host[:-4]
    params = {}
    if version_id is not None:
        params['VersionId'] = version_id
    if response_headers is not None:
        params.update(response_headers)
    http_request = self.build_base_http_request(method, path, auth_path, headers=headers, host=host, params=params)
    return self._auth_handler.presign(http_request, expires_in, iso_date=iso_date)