import pathlib
from base64 import b64encode
from io import BytesIO
import numpy as np
import pytest
from panel.pane import Audio, Video
from panel.pane.media import TensorLike, _is_1dim_int_or_float_tensor
@scipy_available
def test_audio_array(document, comm):
    data = np.random.randint(-100, 100, 100).astype('int16')
    sample_rate = 10
    buffer = BytesIO()
    wavfile.write(buffer, sample_rate, data)
    b64_encoded = b64encode(buffer.getvalue()).decode('utf-8')
    audio = Audio(data, sample_rate=sample_rate)
    model = audio.get_root(document, comm=comm)
    model_value = model.value
    assert model_value.split(',')[1] == b64_encoded
    assert model.value.startswith('data:audio/wav;base64')