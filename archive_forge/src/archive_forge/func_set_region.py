import boto
import boto.ec2
from boto.mashups.server import Server, ServerSet
from boto.mashups.iobject import IObject
from boto.pyami.config import Config
from boto.sdb.persist import get_domain, set_domain
import time
from boto.compat import StringIO
def set_region(self, region=None):
    if region:
        self.region = region
    else:
        l = [(r, r.name, r.endpoint) for r in boto.ec2.regions()]
        self.region = self.choose_from_list(l, prompt='Choose Region')