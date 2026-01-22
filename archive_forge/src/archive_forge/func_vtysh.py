import logging
import os
import netaddr
from . import docker_base as base
def vtysh(self, cmd, config=True):
    if not isinstance(cmd, list):
        cmd = [cmd]
    cmd = ' '.join(("-c '{0}'".format(c) for c in cmd))
    if config:
        return self.exec_on_ctn("vtysh -d bgpd -c 'en' -c 'conf t' -c 'router bgp {0}' {1}".format(self.asn, cmd), capture=True)
    else:
        return self.exec_on_ctn('vtysh -d bgpd {0}'.format(cmd), capture=True)