import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_list_port(self, line):
    """list_port <peer>
        """

    def f(p, args):
        o = p.get()
        for p in o.resources.port:
            print('%s %s %s' % (p.resource_id, p.name, p.number))
    self._request(line, f)