from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@classmethod
def from_download(cls, url, **kwargs):
    """Create a `AudioStream` from a url by downloading

        Parameters
        ----------
        url: str
            The url of the file that will be downloadeded and its bytes
            assigned to the value trait of the video trait.
        **kwargs
            Extra keyword arguments for `AudioStream`
        """
    ext = os.path.splitext(url)[1]
    if ext:
        format = ext[1:]
    audio = Audio(value=urlopen(url).read(), format=format, autoplay=False, controls=False)
    return cls(audio=audio, **kwargs)