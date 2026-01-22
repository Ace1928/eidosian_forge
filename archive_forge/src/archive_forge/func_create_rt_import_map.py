from os_ken.services.protocols.bgp.info_base.vrf import VrfRtImportMap
from os_ken.services.protocols.bgp.info_base.vrf4 import Vrf4NlriImportMap
from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6NlriImportMap
def create_rt_import_map(self, name, value):
    self._create_import_map_factory(name, value, VrfRtImportMap)