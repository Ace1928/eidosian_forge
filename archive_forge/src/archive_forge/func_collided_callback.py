import unittest
import pygame
from pygame import sprite
def collided_callback(spr_a, spr_b, arg_dict_a=arg_dict_a, arg_dict_b=arg_dict_b, return_container=return_container):
    count = arg_dict_a.get(spr_a, 0)
    arg_dict_a[spr_a] = 1 + count
    count = arg_dict_b.get(spr_b, 0)
    arg_dict_b[spr_b] = 1 + count
    return return_container[0]