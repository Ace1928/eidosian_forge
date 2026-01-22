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
@classmethod
def create_from_volume_id(cls, region_name, volume_id, name):
    vol = None
    ec2 = boto.ec2.connect_to_region(region_name)
    rs = ec2.get_all_volumes([volume_id])
    if len(rs) == 1:
        v = rs[0]
        vol = cls()
        vol.volume_id = v.id
        vol.name = name
        vol.region_name = v.region.name
        vol.zone_name = v.zone
        vol.put()
    return vol