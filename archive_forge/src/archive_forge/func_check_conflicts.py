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
def check_conflicts(self, name, msg):
    self.verify_ports()
    if name in self.bridges:
        vsctl_fatal('%s because a bridge named %s already exists' % (msg, name))
    if name in self.ports:
        vsctl_fatal('%s because a port named %s already exists on bridge %s' % (msg, name, self.ports[name].bridge().name))
    if name in self.ifaces:
        vsctl_fatal('%s because an interface named %s already exists on bridge %s' % (msg, name, self.ifaces[name].port().bridge().name))