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
def def_frame(self, frame, *specs):
    names = {'DIRECTION': 'direction', 'HARDWARE-OBJECT': 'hardware_object', 'INITIAL-FREQUENCY': 'initial_frequency', 'SAMPLE-RATE': 'sample_rate', 'CENTER-FREQUENCY': 'center_frequency', 'ENABLE-RAW-CAPTURE': 'enable_raw_capture', 'CHANNEL-DELAY': 'channel_delay'}
    options = {}
    for spec_name, spec_value in specs:
        name = names.get(spec_name, None)
        if name:
            options[name] = json.loads(str(spec_value))
        else:
            raise ValueError(f'Unexpectected attribute {spec_name} in definition of frame {frame}. {frame}, {specs}')
    f = DefFrame(frame, **options)
    return f