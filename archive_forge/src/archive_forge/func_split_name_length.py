from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def split_name_length(name):
    """Split name and length from a name like CubeYF_8"""
    split = name.split('_')
    return (split[0], int(split[1]))