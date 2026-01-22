from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
def get_link_map(self, nslave):
    """
        get the link map, this is a bit hacky, call for better algorithm
        to place similar nodes together
        """
    tree_map, parent_map = self.get_tree(nslave)
    ring_map = self.get_ring(tree_map, parent_map)
    rmap = {0: 0}
    k = 0
    for i in range(nslave - 1):
        k = ring_map[k][1]
        rmap[k] = i + 1
    ring_map_ = {}
    tree_map_ = {}
    parent_map_ = {}
    for k, v in ring_map.items():
        ring_map_[rmap[k]] = (rmap[v[0]], rmap[v[1]])
    for k, v in tree_map.items():
        tree_map_[rmap[k]] = [rmap[x] for x in v]
    for k, v in parent_map.items():
        if k != 0:
            parent_map_[rmap[k]] = rmap[v]
        else:
            parent_map_[rmap[k]] = -1
    return (tree_map_, parent_map_, ring_map_)