import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
def vrrp_shutdown_request(self, ev):
    vrrp_router = self.vrrp_router
    vrrp_router.logger.debug('%s vrrp_shutdown_request', self.__class__.__name__)
    vrrp_router.preempt_delay_timer.cancel()
    vrrp_router.master_down_timer.cancel()
    vrrp_router.state_change(vrrp_event.VRRP_STATE_INITIALIZE)