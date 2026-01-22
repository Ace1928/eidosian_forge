import abc
from abc import ABC
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers as H, mc
from minerl.herobraine.hero.mc import ALL_ITEMS, INVERSE_KEYMAP, SIMPLE_KEYBOARD_ACTION
from minerl.herobraine.env_spec import EnvSpec
from typing import List
import numpy as np
def create_actionables(self) -> List[TranslationHandler]:
    """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
    return [H.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items() if k in SIMPLE_KEYBOARD_ACTION] + [H.CameraAction()]