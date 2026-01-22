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
def create_from_latest_snapshot(self, name, size=None):
    snapshot = self.get_snapshots()[-1]
    return self.create_from_snapshot(name, snapshot, size)