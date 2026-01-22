import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
def vrrp_received(self, ev):
    vrrp_router = self.vrrp_router
    vrrp_router.logger.debug('%s vrrp_received', self.__class__.__name__)
    _ip, vrrp_ = vrrp.vrrp.get_payload(ev.packet)
    if vrrp_.priority == 0:
        vrrp_router.master_down_timer.start(vrrp_router.params.skew_time)
    else:
        params = vrrp_router.params
        config = vrrp_router.config
        if not config.preempt_mode or config.priority <= vrrp_.priority:
            params.master_adver_interval = vrrp_.max_adver_int_in_sec
            vrrp_router.master_down_timer.start(params.master_down_interval)
        elif config.preempt_mode and config.preempt_delay > 0 and (config.priority > vrrp_.priority):
            if not vrrp_router.preempt_delay_timer.is_running():
                vrrp_router.preempt_delay_timer.start(config.preempt_delay)
            vrrp_router.master_down_timer.start(params.master_down_interval)