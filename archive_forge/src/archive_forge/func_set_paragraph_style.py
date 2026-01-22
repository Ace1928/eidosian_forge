from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def set_paragraph_style(self, start, end, attributes):
    return super().set_paragraph_style(0, len(self.text), attributes)