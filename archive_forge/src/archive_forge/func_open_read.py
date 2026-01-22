import base64
import binascii
import os
import re
from boto.compat import StringIO, six
from boto.exception import BotoClientError
from boto.s3.key import Key as S3Key
from boto.s3.keyfile import KeyFile
from boto.utils import compute_hash, get_utf8able_str
def open_read(self, headers=None, query_args='', override_num_retries=None, response_headers=None):
    """
        Open this key for reading

        :type headers: dict
        :param headers: Headers to pass in the web request

        :type query_args: string
        :param query_args: Arguments to pass in the query string
            (ie, 'torrent')

        :type override_num_retries: int
        :param override_num_retries: If not None will override configured
            num_retries parameter for underlying GET.

        :type response_headers: dict
        :param response_headers: A dictionary containing HTTP
            headers/values that will override any headers associated
            with the stored object in the response.  See
            http://goo.gl/EWOPb for details.
        """
    if self.generation:
        if query_args:
            query_args += '&'
        query_args += 'generation=%s' % self.generation
    super(Key, self).open_read(headers=headers, query_args=query_args, override_num_retries=override_num_retries, response_headers=response_headers)