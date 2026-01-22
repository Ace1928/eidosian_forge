import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
def get_animation_decoders(self, filename=None):
    """Get a list of animation decoders. If a `filename` is provided, only
           decoders supporting that extension will be returned. An empty list
           will be return if no encoders for that extension are available.
        """
    if filename:
        extension = os.path.splitext(filename)[1].lower()
        return self._decoder_animation_extensions.get(extension, [])
    return self._decoders