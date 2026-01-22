import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
def get_crafting_action(self):
    """
        :return: action to be taken
        """
    if len(self.crafting_actions) == 0:
        return {}
    result = self.crafting_actions[self.current_action_index]
    self.current_action_index = (self.current_action_index + 1) % len(self.crafting_actions)
    return result