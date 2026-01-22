import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
def getUniqueString(self, prefix=None):
    """Generate unique resource name"""
    return (prefix if prefix else '') + '{time}-{uuid}'.format(time=int(time.time()), uuid=uuid.uuid4().hex)