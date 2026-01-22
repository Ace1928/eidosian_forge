from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def sflow_poll_interval(self):
    if self.want.sflow_poll_interval is None:
        return None
    if self.want.sflow_poll_interval != self.have.sflow_poll_interval:
        return dict(pollInterval=self.want.sflow_poll_interval)