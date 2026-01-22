import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_set_queue_config(self, line):
    """set_queue_config <peer> <target> <queue> <key> <value>
        eg. set_queue_config sw1 running LogicalSwitch7-Port1-Queue922 max-rate 100
        """

    def f(p, args):
        try:
            target, queue, key, value = args
        except:
            print('argument error')
            print(args)
            return
        o = p.get()
        capable_switch_id = o.id
        try:
            capable_switch = ofc.OFCapableSwitchType(id=capable_switch_id, resources=ofc.OFCapableSwitchResourcesType(queue=[ofc.OFQueueType(resource_id=queue, properties=ofc.OFQueuePropertiesType(**{key: value}))]))
        except TypeError:
            print('argument error')
            return
        try:
            p.edit_config(target, capable_switch)
        except Exception as e:
            print(e)
    self._request(line, f)