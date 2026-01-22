import operator
import math
def huslp_to_hex(h, s, l):
    return rgb_to_hex(huslp_to_rgb(h, s, l))