from boto.pyami.config import Config, BotoConfigLocations
from boto.storage_uri import BucketStorageUri, FileStorageUri
import boto.plugin
import datetime
import os
import platform
import re
import sys
import logging
import logging.config
from boto.compat import urlparse
from boto.exception import InvalidUriError
def connect_walrus(host=None, aws_access_key_id=None, aws_secret_access_key=None, port=8773, path='/services/Walrus', is_secure=False, **kwargs):
    """
    Connect to a Walrus service.

    :type host: string
    :param host: the host name or ip address of the Walrus server

    :type aws_access_key_id: string
    :param aws_access_key_id: Your AWS Access Key ID

    :type aws_secret_access_key: string
    :param aws_secret_access_key: Your AWS Secret Access Key

    :rtype: :class:`boto.s3.connection.S3Connection`
    :return: A connection to Walrus
    """
    from boto.s3.connection import S3Connection
    from boto.s3.connection import OrdinaryCallingFormat
    if not aws_access_key_id:
        aws_access_key_id = config.get('Credentials', 'euca_access_key_id', None)
    if not aws_secret_access_key:
        aws_secret_access_key = config.get('Credentials', 'euca_secret_access_key', None)
    if not host:
        host = config.get('Boto', 'walrus_host', None)
    return S3Connection(aws_access_key_id, aws_secret_access_key, host=host, port=port, path=path, calling_format=OrdinaryCallingFormat(), is_secure=is_secure, **kwargs)