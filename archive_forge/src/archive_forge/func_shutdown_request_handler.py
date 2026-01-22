import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
@handler.set_ev_cls(vrrp_event.EventVRRPShutdownRequest)
def shutdown_request_handler(self, ev):
    self._proxy_event(ev)