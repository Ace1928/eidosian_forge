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
@staticmethod
def replace_with_name(name):
    """
        replace all names with human readable variants
        :param name: crafting action or item (item string will be skipped)
        :return: human readable name
        """
    if len(name.split(':')) == 3:
        name, order, digit = name.split(':')
        name = name + ':' + order
        translate = {'place': ['cobblestone', 'crafting_table', 'dirt', 'furnace', 'none', 'stone', 'torch'], 'nearbySmelt': ['coal', 'iron_ingot', 'none'], 'nearbyCraft': ['furnace', 'iron_axe', 'iron_pickaxe', 'none', 'stone_axe', 'stone_pickaxe', 'wooden_axe', 'wooden_pickaxe'], 'equip': ['air', 'iron_axe', 'iron_pickaxe', 'none', 'stone_axe', 'stone_pickaxe', 'wooden_axe', 'wooden_pickaxe'], 'craft': ['crafting_table', 'none', 'planks', 'stick', 'torch']}
        name_without_digits = name
        while name_without_digits not in translate:
            name_without_digits = name_without_digits[:-1]
        return name + ' -> ' + translate[name_without_digits][int(digit)]
    else:
        return name