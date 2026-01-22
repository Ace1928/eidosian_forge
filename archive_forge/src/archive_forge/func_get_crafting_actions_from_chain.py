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
def get_crafting_actions_from_chain(chain_, node_name_):
    """
    getting crafting actions from chain for node_name_ item
    :param chain_:
    :param node_name_: item
    :return:
    """
    previous_actions = []
    for vertex in chain_:
        if vertex == node_name_:
            break
        if not is_item(vertex):
            previous_actions.append(vertex)
        else:
            previous_actions = []
    return [str_to_action_dict(action_) for action_ in previous_actions]