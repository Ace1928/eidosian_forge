import boto
import boto.ec2
from boto.mashups.server import Server, ServerSet
from boto.mashups.iobject import IObject
from boto.pyami.config import Config
from boto.sdb.persist import get_domain, set_domain
import time
from boto.compat import StringIO
def set_ami(self, ami=None):
    if ami:
        self.ami = ami
    else:
        l = [(a, a.id, a.location) for a in self.ec2.get_all_images()]
        self.ami = self.choose_from_list(l, prompt='Choose AMI')