import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def replay2env(self, replay_action, next_action):
    self.last_action = replay_action
    ac = mc.minerec_to_minerl_action(replay_action, next_action=next_action, gui_camera_scaler=self.gui_camera_scaler, esc_to_inventory=False)
    if self.multiagent:
        ac = {'agent_0': ac}
    return ac