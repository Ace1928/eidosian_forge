import pygame
def bitswap(num):
    val = 0
    for x in range(8):
        b = num & 1 << x != 0
        val = val << 1 | b
    return val