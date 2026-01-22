import unittest
import time
from nose.plugins.attrib import attr
from boto.redshift.layer1 import RedshiftConnection
from boto.redshift.exceptions import ClusterNotFoundFault
from boto.redshift.exceptions import ResizeNotFoundFault
def cluster_id(self):
    return self.cluster_prefix % str(int(time.time()))