import os
import re
import warnings
import boto
from boto.compat import expanduser, ConfigParser, NoOptionError, NoSectionError, StringIO
def load_credential_file(self, path):
    """Load a credential file as is setup like the Java utilities"""
    c_data = StringIO()
    c_data.write('[Credentials]\n')
    for line in open(path, 'r').readlines():
        c_data.write(line.replace('AWSAccessKeyId', 'aws_access_key_id').replace('AWSSecretKey', 'aws_secret_access_key'))
    c_data.seek(0)
    self.readfp(c_data)