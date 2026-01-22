import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def simplify_usage(u):
    simplerows = [x.lower().replace(' ', '_') for x in rows]
    setattr(u, simplerows[0], '%d' % len(u.server_usages))
    setattr(u, simplerows[1], '%.2f' % u.total_memory_mb_usage)
    setattr(u, simplerows[2], '%.2f' % u.total_vcpus_usage)
    setattr(u, simplerows[3], '%.2f' % u.total_local_gb_usage)