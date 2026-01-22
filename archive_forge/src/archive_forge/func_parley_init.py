import copy
import random
from typing import List, Dict, Union
from parlai.core.agents import create_agents_from_shared
from parlai.core.loader import load_task_module, load_world_module
from parlai.core.metrics import aggregate_named_reports
from parlai.core.opt import Opt
from parlai.core.teachers import Teacher, create_task_agent_from_taskname
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import Timer, display_messages
from parlai.tasks.tasks import ids_to_tasks
import parlai.utils.logging as logging
def parley_init(self):
    """
        Update the current subworld.

        If we are in the middle of an episode, keep the same world and finish this
        episode. If we have finished this episode, pick a new world (either in a random
        or round-robin fashion).
        """
    self.parleys = self.parleys + 1
    if self.world_idx >= 0 and self.worlds[self.world_idx].episode_done():
        self.new_world = True
    if self.new_world:
        self.new_world = False
        self.parleys = 0
        if self.is_training:
            self.world_idx = random.choices(self.task_choices, cum_weights=self.cum_task_weights)[0]
        else:
            for _ in range(len(self.worlds)):
                self.world_idx = (self.world_idx + 1) % len(self.worlds)
                if not self.worlds[self.world_idx].epoch_done():
                    break