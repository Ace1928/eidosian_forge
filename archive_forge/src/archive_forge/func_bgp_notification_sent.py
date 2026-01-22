from os_ken.services.protocols.bgp.signals import SignalBus
def bgp_notification_sent(self, peer, notification):
    return self.emit_signal(self.BGP_NOTIFICATION_SENT + (peer,), notification)