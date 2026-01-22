from __future__ import print_function
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, ListProperty, ReferenceProperty, CalculatedProperty
from boto.manage.server import Server
from boto.manage import propget
import boto.utils
import boto.ec2
import time
import traceback
from contextlib import closing
import datetime
def get_ec2_connection(self):
    if self.server:
        return self.server.ec2
    if not hasattr(self, 'ec2') or self.ec2 is None:
        self.ec2 = boto.ec2.connect_to_region(self.region_name)
    return self.ec2