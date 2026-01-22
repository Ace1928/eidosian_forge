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
def connect_ec2_endpoint(url, aws_access_key_id=None, aws_secret_access_key=None, **kwargs):
    """
    Connect to an EC2 Api endpoint.  Additional arguments are passed
    through to connect_ec2.

    :type url: string
    :param url: A url for the ec2 api endpoint to connect to

    :type aws_access_key_id: string
    :param aws_access_key_id: Your AWS Access Key ID

    :type aws_secret_access_key: string
    :param aws_secret_access_key: Your AWS Secret Access Key

    :rtype: :class:`boto.ec2.connection.EC2Connection`
    :return: A connection to Eucalyptus server
    """
    from boto.ec2.regioninfo import RegionInfo
    purl = urlparse(url)
    kwargs['port'] = purl.port
    kwargs['host'] = purl.hostname
    kwargs['path'] = purl.path
    if not 'is_secure' in kwargs:
        kwargs['is_secure'] = purl.scheme == 'https'
    kwargs['region'] = RegionInfo(name=purl.hostname, endpoint=purl.hostname)
    kwargs['aws_access_key_id'] = aws_access_key_id
    kwargs['aws_secret_access_key'] = aws_secret_access_key
    return connect_ec2(**kwargs)