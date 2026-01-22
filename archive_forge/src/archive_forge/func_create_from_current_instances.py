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
def create_from_current_instances(cls):
    servers = []
    regions = boto.ec2.regions()
    for region in regions:
        ec2 = region.connect()
        rs = ec2.get_all_reservations()
        for reservation in rs:
            for instance in reservation.instances:
                try:
                    next(Server.find(instance_id=instance.id))
                    boto.log.info('Server for %s already exists' % instance.id)
                except StopIteration:
                    s = cls()
                    s.ec2 = ec2
                    s.name = instance.id
                    s.region_name = region.name
                    s.instance_id = instance.id
                    s._reservation = reservation
                    s.put()
                    servers.append(s)
    return servers