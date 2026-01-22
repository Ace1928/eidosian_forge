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
def neighbor_coords(self, cell, max_size):
    x, y = cell
    X = Y = max_size
    return [(x2, y2) for x2 in range(x - 1, x + 2) for y2 in range(y - 1, y + 2) if -1 < x < X and -1 < y < Y and (x != x2 or y != y2) and (0 <= x2 < X) and (0 <= y2 < Y)]