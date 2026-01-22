import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_get_logical_switch_config(self, line):
    """get_logical_switch_config <peer> <source> <logical switch>
        """

    def f(p, args):
        try:
            source, lsw = args
        except:
            print('argument error')
            return
        o = p.get_config(source)
        for l in o.logical_switches.switch:
            if l.id != lsw:
                continue
            print(l.id)
            for k in self._lsw_settings:
                try:
                    v = getattr(l, k)
                except AttributeError:
                    continue
                print('%s %s' % (k, v))
    self._request(line, f)