from ctypes import *
from ctypes.util import find_library
import os
def router_end(self):
    if self.router is not None:
        if self.router.cmd_rule is None:
            return
        if fluid_midi_router_add_rule(self.router, self.router.cmd_rule, self.router.cmd_rule_type) < 0:
            delete_fluid_midi_router_rule(self.router.cmd_rule)
        self.router.cmd_rule = None