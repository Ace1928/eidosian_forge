import time
import pygame as pg
from pygame.math import Vector2
from typing import List, Set, Tuple, Optional
from random import randint
import numpy as np
import logging
def astar(self, start, end):
    start_node = Node(start['x'], start['y'])
    end_node = Node(end['x'], end['y'])
    open_list = []
    closed_list = []
    open_list.append(start_node)
    possible_paths = []
    adjacent_squares = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    grid_width = screen_size[0] // block_size
    grid_height = screen_size[1] // block_size
    while len(open_list) > 0:
        current_node = open_list[0]
        current_index = 0
        index = 0
        for i in range(len(open_list)):
            if open_list[i].f > current_node.f:
                current_node = open_list[i]
                current_index = index
            index += 1
        open_list.pop(current_index)
        closed_list.append(current_node)
        if current_node.is_equal(end_node):
            path = []
            current = current_node
            while current is not None:
                path.append(current)
                current = current.parent
            possible_paths.append(list(reversed(path)))
        children = []
        for i in range(len(adjacent_squares)):
            node_position = [int(current_node.position.x + adjacent_squares[i][0]), int(current_node.position.y + adjacent_squares[i][1])]
            if 0 <= node_position[0] < grid_width and 0 <= node_position[1] < grid_height:
                new_node = Node(node_position[0], node_position[1])
                children.append(new_node)
        for i in range(len(children)):
            if_in_closed_list = False
            for j in range(len(closed_list)):
                if children[i].is_equal(closed_list[j]):
                    if_in_closed_list = True
            if not if_in_closed_list:
                children[i].g = current_node.g + 2
                children[i].h = abs(children[i].position.x - end_node.position.x) + abs(children[i].position.y - end_node.position.y)
                children[i].f = children[i].g + children[i].h
                present = False
                for j in range(len(open_list)):
                    if children[i].is_equal(open_list[j]) and children[i].g < open_list[j].g:
                        present = True
                    elif children[i].is_equal(open_list[j]) and children[i].g >= open_list[j].g:
                        open_list[j] = children[i]
                        open_list[j].parent = current_node
                if not present:
                    children[i].parent = current_node
                    open_list.append(children[i])
    path = []
    for i in range(len(possible_paths)):
        if len(possible_paths[i]) > len(path):
            path = possible_paths[i]
    return path