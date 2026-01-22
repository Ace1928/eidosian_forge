from os_ken.base import app_manager
from os_ken.services.protocols.vrrp import event as vrrp_event
def vrrp_config(app, interface, config):
    """create an instance.
    returns EventVRRPConfigReply(instance.name, interface, config)
    on success.
    returns EventVRRPConfigReply(None, interface, config)
    on failure.
    """
    config_request = vrrp_event.EventVRRPConfigRequest(interface, config)
    config_request.sync = True
    return app.send_request(config_request)