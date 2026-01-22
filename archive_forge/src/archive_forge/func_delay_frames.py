import json
import pkgutil
import operator
from typing import List
from deprecated import deprecated
from deprecated.sphinx import versionadded
from lark import Lark, Transformer, v_args
import numpy as np
from pyquil.quilbase import (
from pyquil.quiltwaveforms import _wf_from_dict
from pyquil.quilatom import (
from pyquil.gates import (
@v_args(inline=True)
def delay_frames(self, qubit, *frames_and_delay_amount):
    *frame_names, delay_amount = frames_and_delay_amount
    frames = [Frame([qubit], name) for name in frame_names]
    d = DELAY(*[*frames, delay_amount])
    return d