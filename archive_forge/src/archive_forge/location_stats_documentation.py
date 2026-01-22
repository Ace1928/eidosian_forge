import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np

    Includes the current biome, how likely rain and snow are there, as well as the current light level, how bright the
    sky is, and if the player can see the sky.

    Also includes x, y, z, roll, and pitch
    