import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
@lru_cache(maxsize=128)
def format_note_token(self, instrument_bin: int, note: int, velocity_bin: int) -> str:
    return f'{self.cfg.short_instr_bin_names[instrument_bin]}:{note:x}:{velocity_bin:x}'