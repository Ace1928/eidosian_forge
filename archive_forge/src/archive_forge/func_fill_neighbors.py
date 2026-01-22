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
def fill_neighbors(self, grid, size):
    for i in range(size):
        for j in range(size):
            if grid[i, j] == -1:
                neighbors = self.neighbor_coords((i, j), size)
                neighbor_sum = sum((grid[k, l] for k, l in neighbors))
                grid[i, j] = neighbor_sum / len(neighbors)
    return grid