from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import ActivityException
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborsConf
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfsConf
@property
def vrfs_conf(self):
    self._check_started()
    return self._vrfs_conf