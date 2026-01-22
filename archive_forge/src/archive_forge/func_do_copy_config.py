import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_copy_config(self, line):
    """copy_config <peer> <source> <target>
        eg. copy_config sw1 running startup
        """

    def f(p, args):
        try:
            source, target = args
        except:
            print('argument error')
            return
        print(p.copy_config(source, target))
    self._request(line, f)