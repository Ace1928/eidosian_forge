import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
def find_bridge_by_id(self, datapath_id, must_exist):
    assert self.cache_valid
    for vsctl_bridge in self.bridges.values():
        if vsctl_bridge.br_cfg.datapath_id[0].strip('"') == datapath_id:
            self.verify_bridges()
            return vsctl_bridge
    if must_exist:
        vsctl_fatal('no bridge id %s' % datapath_id)
    return None