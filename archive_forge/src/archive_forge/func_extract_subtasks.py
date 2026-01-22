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
@classmethod
def extract_subtasks(cls, trajectory, excluded_actions=('attack', 'back', 'camera', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint'), item_appear_limit=4) -> List[Item]:
    """
        computes item and actions order in time order
        :param trajectory:
        :param excluded_actions: by default all POV actions is excluded
        :param item_appear_limit: filter item vertexes appeared more then item_appear_limit times
        :return:
        """
    states, actions, rewards, next_states, _ = trajectory
    for index in range(len(rewards)):
        for action in actions:
            if action not in excluded_actions:
                a = Action(name=action, value=actions[action][index])
    items = states.keys()
    empty_item = Item(name='empty', value=0, begin=-1, end=0)
    result: List[Item] = [empty_item]
    for index in range(len(rewards)):
        for action in actions:
            "\n                action:  'forward', 'left', 'back', 'right', 'jump', 'sneak', 'sprint', 'attack', 'camera', 'place', 'equip', \n                         'craft', 'nearbyCraft', 'nearbySmelt'\n                "
            if action not in excluded_actions:
                a = Action(name=action, value=actions[action][index])
                last_item = result[-1]
                if not a.is_noop():
                    if last_item.get_last_action() != a:
                        last_item.add_action(a)
        for item in items:
            "\n                items:  odict_keys(['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_axe', \n                                    'iron_ingot', 'iron_ore', 'iron_pickaxe', 'log', 'planks', 'stick', 'stone', \n                                    'stone_axe', 'stone_pickaxe', 'torch', 'wooden_axe', 'wooden_pickaxe'])\n                "
            if next_states[item][index] > states[item][index]:
                i = Item(item, next_states[item][index], begin=result[-1].end, end=index)
                last_item = result[-1]
                if i.name == last_item.name:
                    last_item.value = i.value
                    last_item.end = index
                else:
                    pass
                    result.append(i)
    result.append(empty_item)
    for item, next_item in zip(reversed(result[:-1]), reversed(result[1:])):
        item.actions, next_item.actions = (next_item.actions, item.actions)
    to_remove = set()
    for index, item in enumerate(result):
        if item.begin == item.end:
            to_remove.add(index)
            if index - 1 >= 0:
                to_remove.add(index - 1)
        if sum([1 for _ in result[:index + 1] if _.name == item.name]) >= item_appear_limit:
            to_remove.add(index)
    for index in reversed(sorted(list(to_remove))):
        if result[index].actions:
            result[index + 1].actions = (*result[index].actions, *result[index + 1].actions)
        result.pop(index)
    result = [item for item in result if item != empty_item]
    return result