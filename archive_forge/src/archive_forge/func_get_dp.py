from os_ken.lib.packet import ethernet
from os_ken.lib.packet import vlan
from os_ken.ofproto import ether
from os_ken.topology import api as topo_api
def get_dp(app, dpid):
    """
    :type dpid: datapath id
    :param dpid:
    :rtype: os_ken.controller.controller.Datapath
    :returns: datapath corresponding to dpid
    """
    switches = topo_api.get_switch(app, dpid)
    if not switches:
        return None
    assert len(switches) == 1
    return switches[0].dp