from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def map_regions(self, regions):
    """Draw the key regions onto surface regions.

        Regions must have at least 3 byte pixels. Each pixel of the keyboard
        rectangle is set to the color (note, velocity, 0). The regions surface
        must be at least as large as (0, 0, self.rect.left, self.rect.bottom)

        """
    cutoff = self.black_key_height
    black_keys = []
    for note in range(self._start_note, self._end_note + 1):
        key = self._keys[note]
        if key is not None and key.is_white:
            fill_region(regions, note, key.rect, cutoff)
        else:
            black_keys.append((note, key))
    for note, key in black_keys:
        if key is not None:
            fill_region(regions, note, key.rect, cutoff)