import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def load_actions(self):
    if callable(self.replay_file):
        replay_file = self.replay_file()
    elif isinstance(self.replay_file, str):
        replay_file = self.replay_file
    else:
        raise ValueError('replay_file must be a string or a callable')
    with open(replay_file) as f:
        self.actions = deque([json.loads(l) for l in f.readlines()][:self.max_steps])