from os_ken.base import app_manager
from os_ken.services.protocols.vrrp import event as vrrp_event
def vrrp_config_change(app, instance_name, priority=None, advertisement_interval=None, preempt_mode=None, accept_mode=None):
    """change configuration of an instance.
    None means no change.
    """
    config_change = vrrp_event.EventVRRPConfigChangeRequest(instance_name, priority, advertisement_interval, preempt_mode, accept_mode)
    return app.send_event(vrrp_event.VRRP_MANAGER_NAME, config_change)