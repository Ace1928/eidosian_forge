import boto
import boto.utils
from boto.compat import StringIO
from boto.mashups.iobject import IObject
from boto.pyami.config import Config, BotoConfigPath
from boto.mashups.interactive import interactive_shell
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty
import os
def install_package(self, package_name):
    print('installing %s...' % package_name)
    command = 'yum -y install %s' % package_name
    print('\t%s' % command)
    ssh_client = self.get_ssh_client()
    t = ssh_client.exec_command(command)
    response = t[1].read()
    print('\t%s' % response)
    print('\t%s' % t[2].read())
    print('...complete!')