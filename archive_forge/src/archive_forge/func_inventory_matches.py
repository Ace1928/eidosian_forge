import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def inventory_matches(inv_ob, inv_json):
    inv_dict = defaultdict(int)
    for itemstack in inv_json:
        inv_dict[itemstack['type']] += itemstack['quantity']
    for item, quantity in inv_dict.items():
        if int(inv_ob[item]) != quantity:
            print(f'Inventory mismatch! Item {item}: agent has {inv_ob[item]}, should have {quantity}')
            return False
    return True