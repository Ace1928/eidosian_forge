from __future__ import print_function
import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six
from heapq import merge
from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
from .utilities import (key_name_to_key_number, qpm_to_bpm)
Compares two events for sorting.

            Events are sorted by tick time ascending. Events with the same tick
            time ares sorted by event type. Some events are sorted by
            additional values. For example, Note On events are sorted by pitch
            then velocity, ensuring that a Note Off (Note On with velocity 0)
            will never follow a Note On with the same pitch.

            Parameters
            ----------
            event1, event2 : mido.Message
               Two events to be compared.
            