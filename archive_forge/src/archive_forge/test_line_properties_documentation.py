import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
Asserts that self.func matches as described
        by s, which uses a little language to describe matches:

        abcd<efg>hijklmnopqrstuvwx|yz
           /|\ /|\               /|\
            |   |                 |
         the function should   the current cursor position
         match this "efg"      is between the x and y
        