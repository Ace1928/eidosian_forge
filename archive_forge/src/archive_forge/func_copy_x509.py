import boto.ec2
from boto.mashups.iobject import IObject
from boto.pyami.config import BotoConfigPath, Config
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, BooleanProperty, CalculatedProperty
from boto.manage import propget
from boto.ec2.zone import Zone
from boto.ec2.keypair import KeyPair
import os, time
from contextlib import closing
from boto.exception import EC2ResponseError
from boto.compat import six, StringIO
def copy_x509(self, key_file, cert_file):
    print('\tcopying cert and pk over to /mnt directory on server')
    self.ssh_client.open_sftp()
    path, name = os.path.split(key_file)
    self.remote_key_file = '/mnt/%s' % name
    self.ssh_client.put_file(key_file, self.remote_key_file)
    path, name = os.path.split(cert_file)
    self.remote_cert_file = '/mnt/%s' % name
    self.ssh_client.put_file(cert_file, self.remote_cert_file)
    print('...complete!')