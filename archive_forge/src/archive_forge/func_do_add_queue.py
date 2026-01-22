import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_add_queue(self, line):
    """add_queue <peer> <target> <logical-switch> <queue>
        eg. add_queue sw1 running LogicalSwitch7 NameOfNewQueue
        """

    def f(p, args):
        try:
            target, lsw, queue = args
        except:
            print('argument error')
            print(args)
            return
        o = p.get()
        capable_switch_id = o.id
        try:
            capable_switch = ofc.OFCapableSwitchType(id=capable_switch_id, resources=ofc.OFCapableSwitchResourcesType(queue=[ofc.OFQueueType(resource_id=queue)]), logical_switches=ofc.OFCapableSwitchLogicalSwitchesType(switch=[ofc.OFLogicalSwitchType(id=lsw, resources=ofc.OFLogicalSwitchResourcesType(queue=[queue]))]))
        except TypeError:
            print('argument error')
            return
        try:
            p.edit_config(target, capable_switch)
        except Exception as e:
            print(e)
    self._request(line, f)