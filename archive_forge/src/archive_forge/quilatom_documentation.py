from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
A reference implementation of waveform sample generation.

        Note: this is close but not always exactly equivalent to the actual IQ
        values produced by the waveform generators on Rigetti hardware. The
        actual ADC process imposes some alignment constraints on the waveform
        duration (in particular, it must be compatible with the clock rate).

        :param rate: The sample rate, in Hz.
        :returns: An array of complex samples.

        