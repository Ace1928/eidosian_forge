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
def del_cached_bridge(self, vsctl_bridge):
    assert not vsctl_bridge.ports
    assert not vsctl_bridge.children
    parent = vsctl_bridge.parent
    if parent:
        parent.children.remove(vsctl_bridge)
        vsctl_bridge.parent = None
    ovsrec_bridge = vsctl_bridge.br_cfg
    if ovsrec_bridge:
        ovsrec_bridge.delete()
        self.ovs_delete_bridge(ovsrec_bridge)
    del self.bridges[vsctl_bridge.name]