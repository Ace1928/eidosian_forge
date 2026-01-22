import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
def vrrp_config_change_request(self, ev):
    vrrp_router = self.vrrp_router
    vrrp_router.logger.warning('%s vrrp_config_change_request', self.__class__.__name__)
    if ev.priority is not None and vrrp_router.config.address_owner:
        vrrp_router.master_down_timer.cancel()
        self._master_down()
    if ev.preempt_mode is not None or ev.preempt_delay is not None:
        vrrp_router.preempt_delay_timer.cancel()