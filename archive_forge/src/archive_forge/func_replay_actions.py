import json
import os
import random
import time
import copy
import numpy as np
import pickle
from joblib import Parallel, delayed
from parlai.core.worlds import MultiAgentDialogWorld
from parlai.mturk.core.agents import MTURK_DISCONNECT_MESSAGE
from parlai.mturk.core.worlds import MTurkOnboardWorld
def replay_actions(self):
    """
        Replays a loaded dialog in the mturk interface.
        """
    tourist = self.agents[0]
    guide = self.agents[1]
    cur_time = None
    actions = []
    start = self.start_idx
    time.sleep(5)
    for i in range(start, len(self.replay_acts)):
        act = self.replay_acts[i]
        if self.real_time:
            if cur_time is None:
                cur_time = act['time']
            else:
                elapsed = act['time'] - cur_time
                if not self.is_action(elapsed):
                    elapsed *= 0.75
                    if not self.real_time:
                        elapsed = min(elapsed, 2)
                time.sleep(elapsed)
                cur_time = act['time']
        else:
            time.sleep(2)
        if self.is_action(act['text']):
            self.update_location(act['text'])
            act['id'] = 'ACTION'
            tourist.observe(act)
            act['id'] = 'Tourist'
            actions.append(act)
            continue
        if act['text'] == 'EVALUATE_LOCATION':
            done = self.evaluate_location()
            if done:
                self.episodeDone = True
                return
        else:
            if self.replay_bot:
                if act['id'] == 'Tourist' and self.bot_type != 'natural':
                    text = act['text']
                    act['text'] = text[:16]
                elif act['id'] == 'Guide':
                    grid = act['text']
                    old_grid = np.array(grid)
                    sizes = [9, 19, 39]
                    for i in sizes:
                        new_grid = self.construct_expanded_array(old_grid, i)
                        old_grid = new_grid
                    act['attn_grid'] = new_grid[:37, :37].tolist()
                    act['attn_grid_size'] = sizes[-1] - 2
                    binary_grid = ''
                    mean = np.mean(np.array(grid))
                    for i in range(len(grid)):
                        for j in range(len(grid[i])):
                            num = int(grid[i][j] > mean)
                            binary_grid += str(num)
                    act['show_grid'] = True
                    act['text'] = binary_grid
            guide.observe(act)
            if 'attn_grid' in act:
                act['attn_grid'] == []
            tourist.observe(act)
    self.episodeDone = True