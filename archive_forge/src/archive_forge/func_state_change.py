import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
def state_change(self, new_state):
    old_state = self.state
    self.state = new_state
    self.state_impl = self._STATE_MAP[new_state](self)
    state_changed = vrrp_event.EventVRRPStateChanged(self.name, self.monitor_name, self.interface, self.config, old_state, new_state)
    self.send_event_to_observers(state_changed)