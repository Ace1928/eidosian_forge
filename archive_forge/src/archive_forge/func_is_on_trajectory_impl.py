import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def is_on_trajectory_impl(self, replay_action, ob, info):
    max_dcoord = self.max_dcoord
    location_stats = info['location_stats']
    x = location_stats['xpos']
    y = location_stats['ypos']
    z = location_stats['zpos']
    yaw = location_stats['yaw']
    pitch = location_stats['pitch']
    x1 = replay_action['xpos']
    y1 = replay_action['ypos']
    z1 = replay_action['zpos']
    tick1 = replay_action['tick']
    yaw1 = replay_action['yaw']
    pitch1 = replay_action['pitch']
    if abs(x - x1) > max_dcoord or abs(y - y1) > max_dcoord or abs(z - z1) > max_dcoord or (abs(yaw - yaw1) > max_dcoord) or (abs(pitch - pitch1) > max_dcoord):
        print(f'Tick {tick1}: Coords mismatch: is {x}, {y}, {z}, {yaw}, {pitch}, should be {x1}, {y1}, {z1}, {yaw1}, {pitch1}')
        self.mismatched_ticks += 1
    elif 'inventory' in replay_action and (not inventory_matches(ob['inventory'], replay_action['inventory'])):
        print(f'Tick {tick1}: Inventory mismatch')
        self.mismatched_ticks += 1
    else:
        self.mismatched_ticks = 0
    return self.mismatched_ticks < self.max_mismatched_ticks