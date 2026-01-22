import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
def get_animation_file_extensions(self):
    """Return a list of accepted file extensions, e.g. ['.gif', '.flc']
        Lower-case only.
        """
    return []