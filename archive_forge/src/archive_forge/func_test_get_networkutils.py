import inspect
from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import rdpconsoleutils
from os_win.utils.compute import vmutils
from os_win.utils.dns import dnsutils
from os_win.utils import hostutils
from os_win.utils.io import ioutils
from os_win.utils.network import networkutils
from os_win.utils import pathutils
from os_win.utils import processutils
from os_win.utils.storage import diskutils
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.storage import smbutils
from os_win.utils.storage.virtdisk import vhdutils
from os_win import utilsfactory
def test_get_networkutils(self):
    self._check_get_class(expected_class=networkutils.NetworkUtils, class_type='networkutils')