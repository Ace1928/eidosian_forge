import re
import unittest
import psutil
from psutil import AIX
from psutil.tests import PsutilTestCase
from psutil.tests import sh
def test_net_if_addrs_names(self):
    out = sh('/etc/ifconfig -l')
    ifconfig_names = set(out.split())
    psutil_names = set(psutil.net_if_addrs().keys())
    self.assertSetEqual(ifconfig_names, psutil_names)