import pathlib
from base64 import b64encode
from io import BytesIO
import numpy as np
import pytest
from panel.pane import Audio, Video
from panel.pane.media import TensorLike, _is_1dim_int_or_float_tensor
@scipy_available
def test_numpy_audio(document, comm):
    sps = 8000
    duration = 0.01
    modulator_frequency = 2.0
    carrier_frequency = 120.0
    modulation_index = 2.0
    time = np.arange(sps * duration) / sps
    modulator = np.sin(2.0 * np.pi * modulator_frequency * time) * modulation_index
    waveform = np.sin(2.0 * np.pi * (carrier_frequency * time + modulator))
    waveform_quiet = waveform * 0.3
    waveform_int = np.int16(waveform_quiet * 32767)
    audio = Audio(waveform_int, sample_rate=sps)
    model = audio.get_root(document, comm=comm)
    assert model.value == 'data:audio/wav;base64,UklGRsQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YaAAAAAAAF4ErQjgDOgQuBRDGH0bXB7WIOMifCScJT8mYyYHJi0l1yMLIs0fJR0dGr4WFBMqDw4LzQZ1Ahf+vfl59VfxZu2z6UrmN+OD4DjeXNz42g7aotm22UjaWNvi3ODeTOEe5E3nzuqU7pXywvYO+2v/yAMZCFAMXhA1FMoXDxv7HYMgoCJJJHolLyZlJhwmVSURJFciKiCTHZoaSBeqE8oP'