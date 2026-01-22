import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def generate_program_change_messages(cfg: VocabConfig):
    for bin_name, channel in cfg.bin_channel_map.items():
        if channel == 9:
            continue
        program = cfg._instrument_names_str_to_int[cfg.bin_name_to_program_name[bin_name]]
        yield mido.Message('program_change', program=program, time=0, channel=channel)
    yield mido.Message('program_change', program=0, time=0, channel=9)