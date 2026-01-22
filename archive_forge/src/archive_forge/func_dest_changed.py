from os_ken.services.protocols.bgp.signals import SignalBus
def dest_changed(self, dest):
    return self.emit_signal(self.BGP_DEST_CHANGED, dest)