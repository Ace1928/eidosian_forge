from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
def on_player_eos(self):
    """The player ran out of sources. The playlist is empty.

        :event:
        """
    if _debug:
        print('Player.on_player_eos')