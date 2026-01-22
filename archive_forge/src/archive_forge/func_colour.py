import re
import sys
from html import escape
from weakref import proxy
from .std import tqdm as std_tqdm
@colour.setter
def colour(self, bar_color):
    if hasattr(self, 'container'):
        self.container.children[-2].style.bar_color = bar_color