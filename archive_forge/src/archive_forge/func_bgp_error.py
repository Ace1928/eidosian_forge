from os_ken.services.protocols.bgp.signals import SignalBus
def bgp_error(self, peer, code, subcode, reason):
    return self.emit_signal(self.BGP_ERROR + (peer,), {'code': code, 'subcode': subcode, 'reason': reason, 'peer': peer})