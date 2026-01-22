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
def is_world_success(self, world):
    """
        Determines whether a given world/dialog yielded a successful run of the task.

        Used when loading a world from data for replay.
        """
    target_location = world['target_location']
    start_location = world['start_location']
    location = start_location
    replay_acts = world['dialog']
    min_x, min_y, max_x, max_y = world['boundaries']
    num_evaluations = 0
    last_grid = None

    def evaluate_location(num_evals, location, target):
        if num_evals == 3:
            return (num_evals, False, True)
        num_evals += 1
        return (num_evals, location[0] == target[0] and location[1] == target[1], False)

    def update_location(act, loc, mi_x, ma_x, mi_y, ma_y):
        if act == 'ACTION:TURNLEFT':
            loc[2] = (loc[2] - 1) % 4
        if act == 'ACTION:TURNRIGHT':
            loc[2] = (loc[2] + 1) % 4
        if act == 'ACTION:FORWARD':
            orientation = self.orientations[loc[2]]
            loc[0] += self.steps[orientation][0]
            loc[1] += self.steps[orientation][1]
            loc[0] = max(min(loc[0], ma_x), mi_x)
            loc[1] = max(min(loc[1], ma_y), mi_y)
        return loc
    for kk, act in enumerate(replay_acts):
        if self.is_action(act['text']):
            location = update_location(act['text'], location, min_x, max_x, min_y, max_y)
        elif act['text'] == 'EVALUATE_LOCATION':
            num_evals, done, too_many = evaluate_location(num_evaluations, location, target_location)
            if done:
                max_prob = 0
                max_i_j = None
                for i in range(len(last_grid)):
                    for j in range(len(last_grid[i])):
                        if last_grid[i][j] > max_prob:
                            max_i_j = (i, j)
                            max_prob = last_grid[i][j]
                if max_i_j != (location[0] - min_x, location[1] - min_y):
                    return (False, -1)
                high_prob = any((any((k >= 0.5 for k in j)) for j in last_grid))
                max_prob = max((max(j) for j in last_grid))
                return (True and high_prob, kk)
            elif too_many:
                return (False, -1)
        elif act['id'] == 'Guide':
            last_grid = act['text']
    return (False, -1)