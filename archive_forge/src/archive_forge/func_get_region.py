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
def get_region(self, params):
    if not params.get('region', None):
        prop = self.cls.find_property('region_name')
        params['region'] = propget.get(prop, choices=boto.ec2.regions)