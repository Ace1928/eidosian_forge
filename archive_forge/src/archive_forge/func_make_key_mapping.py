from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def make_key_mapping(keys, start_note):
    """Return a dictionary of (note, velocity) by computer keyboard key code"""
    mapping = {}
    for i, key in enumerate(keys):
        mapping[key] = (start_note + i, 127)
    return mapping