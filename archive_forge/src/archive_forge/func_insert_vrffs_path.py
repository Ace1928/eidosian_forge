import abc
import logging
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.info_base.vrf import VrfTable
from os_ken.services.protocols.bgp.info_base.vrf import VrfDest
from os_ken.services.protocols.bgp.info_base.vrf import VrfPath
from os_ken.services.protocols.bgp.utils.bgp import create_rt_extended_community
def insert_vrffs_path(self, nlri, communities, is_withdraw=False):
    assert nlri
    assert isinstance(communities, list)
    vrf_conf = self.vrf_conf
    from os_ken.services.protocols.bgp.core import EXPECTED_ORIGIN
    pattrs = OrderedDict()
    pattrs[BGP_ATTR_TYPE_ORIGIN] = BGPPathAttributeOrigin(EXPECTED_ORIGIN)
    pattrs[BGP_ATTR_TYPE_AS_PATH] = BGPPathAttributeAsPath([])
    for rt in vrf_conf.export_rts:
        communities.append(create_rt_extended_community(rt, 2))
    for soo in vrf_conf.soo_list:
        communities.append(create_rt_extended_community(soo, 3))
    pattrs[BGP_ATTR_TYPE_EXTENDED_COMMUNITIES] = BGPPathAttributeExtendedCommunities(communities=communities)
    puid = self.VRF_PATH_CLASS.create_puid(vrf_conf.route_dist, nlri.prefix)
    path = self.VRF_PATH_CLASS(puid, None, nlri, 0, pattrs=pattrs, is_withdraw=is_withdraw)
    eff_dest = self.insert(path)
    self._signal_bus.dest_changed(eff_dest)