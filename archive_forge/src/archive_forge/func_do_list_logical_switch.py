import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_list_logical_switch(self, line):
    """list_logical_switch <peer>
        """

    def f(p, args):
        o = p.get()
        for s in o.logical_switches.switch:
            print('%s %s' % (s.id, s.datapath_id))
    self._request(line, f)