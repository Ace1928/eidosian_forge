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
def build_post_form_args(self, bucket_name, key, expires_in=6000, acl=None, success_action_redirect=None, max_content_length=None, http_method='http', fields=None, conditions=None, storage_class='STANDARD', server_side_encryption=None):
    """
        Taken from the AWS book Python examples and modified for use with boto
        This only returns the arguments required for the post form, not the
        actual form.  This does not return the file input field which also
        needs to be added

        :type bucket_name: string
        :param bucket_name: Bucket to submit to

        :type key: string
        :param key:  Key name, optionally add ${filename} to the end to
            attach the submitted filename

        :type expires_in: integer
        :param expires_in: Time (in seconds) before this expires, defaults
            to 6000

        :type acl: string
        :param acl: A canned ACL.  One of:
            * private
            * public-read
            * public-read-write
            * authenticated-read
            * bucket-owner-read
            * bucket-owner-full-control

        :type success_action_redirect: string
        :param success_action_redirect: URL to redirect to on success

        :type max_content_length: integer
        :param max_content_length: Maximum size for this file

        :type http_method: string
        :param http_method:  HTTP Method to use, "http" or "https"

        :type storage_class: string
        :param storage_class: Storage class to use for storing the object.
            Valid values: STANDARD | REDUCED_REDUNDANCY

        :type server_side_encryption: string
        :param server_side_encryption: Specifies server-side encryption
            algorithm to use when Amazon S3 creates an object.
            Valid values: None | AES256

        :rtype: dict
        :return: A dictionary containing field names/values as well as
            a url to POST to

            .. code-block:: python


        """
    if fields is None:
        fields = []
    if conditions is None:
        conditions = []
    expiration = time.gmtime(int(time.time() + expires_in))
    conditions.append('{"bucket": "%s"}' % bucket_name)
    if key.endswith('${filename}'):
        conditions.append('["starts-with", "$key", "%s"]' % key[:-len('${filename}')])
    else:
        conditions.append('{"key": "%s"}' % key)
    if acl:
        conditions.append('{"acl": "%s"}' % acl)
        fields.append({'name': 'acl', 'value': acl})
    if success_action_redirect:
        conditions.append('{"success_action_redirect": "%s"}' % success_action_redirect)
        fields.append({'name': 'success_action_redirect', 'value': success_action_redirect})
    if max_content_length:
        conditions.append('["content-length-range", 0, %i]' % max_content_length)
    if self.provider.security_token:
        fields.append({'name': 'x-amz-security-token', 'value': self.provider.security_token})
        conditions.append('{"x-amz-security-token": "%s"}' % self.provider.security_token)
    if storage_class:
        fields.append({'name': 'x-amz-storage-class', 'value': storage_class})
        conditions.append('{"x-amz-storage-class": "%s"}' % storage_class)
    if server_side_encryption:
        fields.append({'name': 'x-amz-server-side-encryption', 'value': server_side_encryption})
        conditions.append('{"x-amz-server-side-encryption": "%s"}' % server_side_encryption)
    policy = self.build_post_policy(expiration, conditions)
    policy_b64 = base64.b64encode(policy)
    fields.append({'name': 'policy', 'value': policy_b64})
    fields.append({'name': 'AWSAccessKeyId', 'value': self.aws_access_key_id})
    signature = self._auth_handler.sign_string(policy_b64)
    fields.append({'name': 'signature', 'value': signature})
    fields.append({'name': 'key', 'value': key})
    url = '%s://%s/' % (http_method, self.calling_format.build_host(self.server_name(), bucket_name))
    return {'action': url, 'fields': fields}