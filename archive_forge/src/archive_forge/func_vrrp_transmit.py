from os_ken.base import app_manager
from os_ken.services.protocols.vrrp import event as vrrp_event
def vrrp_transmit(app, monitor_name, data):
    """transmit a packet from the switch.  this is internal use only.
    data is str-like, a packet to send.
    """
    transmit_request = vrrp_event.EventVRRPTransmitRequest(data)
    app.send_event(monitor_name, transmit_request)