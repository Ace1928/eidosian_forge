from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Pattern, Union, Optional, List, Any, Tuple, Callable, Iterator, Type, Dict, \
import pyglet
from pyglet import graphics
from pyglet.customtypes import AnchorX, AnchorY, ContentVAlign, HorizontalAlign
from pyglet.font.base import Font, Glyph
from pyglet.gl import GL_TRIANGLES, GL_LINES, glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, \
from pyglet.image import Texture
from pyglet.text import runlist
from pyglet.text.runlist import RunIterator, AbstractRunIterator
def get_as_texture(self, min_filter=GL_NEAREST, mag_filter=GL_NEAREST) -> Texture:
    """Returns a Texture with the TextLayout drawn to it. Each call to this function returns a new
        Texture.
        ~Warning: Usage is recommended only if you understand how texture generation affects your application.
        """
    framebuffer = pyglet.image.Framebuffer()
    temp_pos = self.position
    width = int(round(self._content_width))
    height = int(round(self._content_height))
    texture = pyglet.image.Texture.create(width, height, min_filter=min_filter, mag_filter=mag_filter)
    depth_buffer = pyglet.image.buffer.Renderbuffer(width, height, GL_DEPTH_COMPONENT)
    framebuffer.attach_texture(texture)
    framebuffer.attach_renderbuffer(depth_buffer, attachment=GL_DEPTH_ATTACHMENT)
    self.position = (0 - self._anchor_left, 0 - self._anchor_bottom, 0)
    framebuffer.bind()
    self.draw()
    framebuffer.unbind()
    self.position = temp_pos
    return texture