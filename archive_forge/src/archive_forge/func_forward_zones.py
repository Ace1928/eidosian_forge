from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def forward_zones(self):
    if self.want.forward_zones is None:
        return None
    if self.have.forward_zones is None and self.want.forward_zones in ['', 'none']:
        return None
    if self.have.forward_zones is not None and self.want.forward_zones in ['', 'none']:
        return []
    if self.have.forward_zones is None:
        return dict(forward_zones=self.want.forward_zones)
    want = sorted(self.want.forward_zones, key=lambda x: x['name'])
    have = sorted(self.have.forward_zones, key=lambda x: x['name'])
    wnames = [x['name'] for x in want]
    hnames = [x['name'] for x in have]
    if set(wnames) != set(hnames):
        return dict(forward_zones=self.want.forward_zones)
    for idx, x in enumerate(want):
        wns = x.get('nameservers', [])
        hns = have[idx].get('nameservers', [])
        if set(wns) != set(hns):
            return dict(forward_zones=self.want.forward_zones)