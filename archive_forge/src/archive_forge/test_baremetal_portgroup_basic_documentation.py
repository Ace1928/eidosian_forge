import ddt
from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
Check baremetal port group set and unset commands.

        Test steps:
        1) Create baremetal port group in setUp.
        2) Set extra data for port group.
        3) Check that baremetal port group extra data was set.
        4) Unset extra data for port group.
        5) Check that baremetal port group extra data was unset.
        