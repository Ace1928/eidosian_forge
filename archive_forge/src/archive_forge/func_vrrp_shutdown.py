from os_ken.base import app_manager
from os_ken.services.protocols.vrrp import event as vrrp_event
def vrrp_shutdown(app, instance_name):
    """shutdown the instance.
    """
    shutdown_request = vrrp_event.EventVRRPShutdownRequest(instance_name)
    app.send_event(vrrp_event.VRRP_MANAGER_NAME, shutdown_request)