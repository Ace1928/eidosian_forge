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
def del_bridge(self, vsctl_bridge):
    for child in vsctl_bridge.children.copy():
        self.del_bridge(child)
    for vsctl_port in vsctl_bridge.ports.copy():
        self.del_port(vsctl_port)
    self.del_cached_bridge(vsctl_bridge)