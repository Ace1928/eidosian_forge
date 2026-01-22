from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
def seek_next_frame(self) -> None:
    """Step forwards one video frame in the current source."""
    time = self.source.get_next_video_timestamp()
    if time is None:
        return
    self.seek(time)