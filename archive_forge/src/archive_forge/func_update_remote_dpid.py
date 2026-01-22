import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
def update_remote_dpid(self, dpid, port_no, remote_dpid):
    remote_dpid_ = self.dpids[dpid].get(port_no)
    if remote_dpid_ is None:
        self._add_remote_dpid(dpid, port_no, remote_dpid)
    elif remote_dpid_ != remote_dpid:
        raise os_ken_exc.RemoteDPIDAlreadyExist(dpid=dpid, port=port_no, remote_dpid=remote_dpid)