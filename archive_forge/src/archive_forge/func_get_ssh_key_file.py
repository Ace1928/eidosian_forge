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
def get_ssh_key_file(self):
    if not self.ssh_key_file:
        ssh_dir = os.path.expanduser('~/.ssh')
        if os.path.isdir(ssh_dir):
            ssh_file = os.path.join(ssh_dir, '%s.pem' % self.key_name)
            if os.path.isfile(ssh_file):
                self.ssh_key_file = ssh_file
        if not self.ssh_key_file:
            iobject = IObject()
            self.ssh_key_file = iobject.get_filename('Path to OpenSSH Key file')
    return self.ssh_key_file