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
@classmethod
def create_from_instance_id(cls, instance_id, name, description=''):
    regions = boto.ec2.regions()
    for region in regions:
        ec2 = region.connect()
        try:
            rs = ec2.get_all_reservations([instance_id])
        except:
            rs = []
        if len(rs) == 1:
            s = cls()
            s.ec2 = ec2
            s.name = name
            s.description = description
            s.region_name = region.name
            s.instance_id = instance_id
            s._reservation = rs[0]
            for instance in s._reservation.instances:
                if instance.id == instance_id:
                    s._instance = instance
            s.put()
            return s
    return None