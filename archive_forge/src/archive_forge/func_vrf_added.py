from os_ken.services.protocols.bgp.signals import SignalBus
def vrf_added(self, vrf_conf):
    return self.emit_signal(self.BGP_VRF_ADDED, vrf_conf)