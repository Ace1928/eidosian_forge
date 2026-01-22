import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, study, ops, value, protocols
from cirq._doc import document
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import InitObsSetting, observables_to_settings, _MeasurementSpec
Call `cirq.to_json with `value` according to the configuration options in this class.

        If `checkpoint=False`, nothing will happen. Otherwise, we will use `checkpoint_fn` and
        `checkpoint_other_fn` as the destination JSON file as described in the class docstring.
        