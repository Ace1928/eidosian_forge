import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter
import importlib
import memory_profiler as mp
def mouse_click_handler(event):
    x, y = (event.xdata, event.ydata)
    if x is None or y is None:
        return
    for coord, _ in rectangles.items():
        x0, y0, x1, y1 = coord
        if x0 < x < x1 and y0 < y < y1:
            toolbar = pl.gcf().canvas.toolbar
            toolbar.push_current()
            timestamp_ax.set_xlim(x0, x1)
            timestamp_ax.set_ylim(y0, stack_size + 1)
            toolbar.push_current()
            pl.draw()
            return