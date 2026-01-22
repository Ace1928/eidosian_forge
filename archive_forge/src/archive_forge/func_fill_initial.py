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
def fill_initial(self, new_g, old_g, size):
    for i in (0, size - 1):
        for j in range(size):
            new_g[i, j] = 0
    for j in (0, size - 1):
        for i in range(size):
            new_g[i, j] = 0
    for i in range(1, size, 2):
        for j in range(1, size, 2):
            new_g[i, j] = old_g[(i - 1) // 2, (j - 1) // 2]
    for i in range(1, size - 1, 2):
        for j in range(2, size - 1, 2):
            new_g[i, j] = (new_g[i, j - 1] + new_g[i, j + 1]) / 2
    for i in range(2, size - 1, 2):
        for j in range(1, size - 1, 2):
            new_g[i, j] = (new_g[i - 1, j] + new_g[i + 1, j]) / 2
    return new_g