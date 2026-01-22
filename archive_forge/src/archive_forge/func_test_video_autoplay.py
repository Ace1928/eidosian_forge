import pathlib
from base64 import b64encode
from io import BytesIO
import numpy as np
import pytest
from panel.pane import Audio, Video
from panel.pane.media import TensorLike, _is_1dim_int_or_float_tensor
def test_video_autoplay(document, comm):
    video = Video(str(ASSETS / 'mp4.mp4'), autoplay=True)
    model = video.get_root(document, comm=comm)
    assert model.autoplay