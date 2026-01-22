import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def handle_note_off(ch, prog, n):
    if channel_pedal_on[ch]:
        channel_pedal_events[ch][n, prog] = True
    else:
        consume_note_program_data(prog, ch, n, 0)
        if n in channel_notes[ch]:
            del channel_notes[ch][n]