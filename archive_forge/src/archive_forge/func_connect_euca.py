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
def connect_euca(host=None, aws_access_key_id=None, aws_secret_access_key=None, port=8773, path='/services/Eucalyptus', is_secure=False, **kwargs):
    """
    Connect to a Eucalyptus service.

    :type host: string
    :param host: the host name or ip address of the Eucalyptus server

    :type aws_access_key_id: string
    :param aws_access_key_id: Your AWS Access Key ID

    :type aws_secret_access_key: string
    :param aws_secret_access_key: Your AWS Secret Access Key

    :rtype: :class:`boto.ec2.connection.EC2Connection`
    :return: A connection to Eucalyptus server
    """
    from boto.ec2 import EC2Connection
    from boto.ec2.regioninfo import RegionInfo
    if not aws_access_key_id:
        aws_access_key_id = config.get('Credentials', 'euca_access_key_id', None)
    if not aws_secret_access_key:
        aws_secret_access_key = config.get('Credentials', 'euca_secret_access_key', None)
    if not host:
        host = config.get('Boto', 'eucalyptus_host', None)
    reg = RegionInfo(name='eucalyptus', endpoint=host)
    return EC2Connection(aws_access_key_id, aws_secret_access_key, region=reg, port=port, path=path, is_secure=is_secure, **kwargs)