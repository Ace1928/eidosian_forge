from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def input_main(device_id=None):
    pg.init()
    pygame.midi.init()
    _print_device_info()
    if device_id is None:
        input_id = pygame.midi.get_default_input_id()
    else:
        input_id = device_id
    print(f'using input_id :{input_id}:')
    i = pygame.midi.Input(input_id)
    pg.display.set_mode((1, 1))
    going = True
    while going:
        events = pygame.event.get()
        for e in events:
            if e.type in [pg.QUIT]:
                going = False
            if e.type in [pg.KEYDOWN]:
                going = False
            if e.type in [pygame.midi.MIDIIN]:
                print(e)
        if i.poll():
            midi_events = i.read(10)
            midi_evs = pygame.midi.midis2events(midi_events, i.device_id)
            for m_e in midi_evs:
                pygame.event.post(m_e)
    del i
    pygame.midi.quit()