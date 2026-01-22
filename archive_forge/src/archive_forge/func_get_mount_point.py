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
def get_mount_point(self, params):
    if not params.get('mount_point', None):
        prop = self.cls.find_property('mount_point')
        params['mount_point'] = propget.get(prop)