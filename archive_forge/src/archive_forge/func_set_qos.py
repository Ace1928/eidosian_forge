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
def set_qos(self, vsctl_port, type, max_rate):
    qos = vsctl_port.qos.qos_cfg
    if not len(qos):
        ovsrec_qos = self.txn.insert(self.txn.idl.tables[vswitch_idl.OVSREC_TABLE_QOS])
        vsctl_port.port_cfg.qos = [ovsrec_qos]
    else:
        ovsrec_qos = qos[0]
    ovsrec_qos.type = type
    if max_rate is not None:
        value_json = ['map', [['max-rate', max_rate]]]
        self.set_column(ovsrec_qos, 'other_config', value_json)
    self.add_qos_to_cache(vsctl_port, [ovsrec_qos])
    return ovsrec_qos