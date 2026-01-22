from kivy.event import EventDispatcher
from kivy.eventmanager import (
from kivy.factory import Factory
from kivy.properties import (
from kivy.graphics import (
from kivy.graphics.transformation import Matrix
from kivy.base import EventLoop
from kivy.lang import Builder
from kivy.context import get_current_context
from kivy.weakproxy import WeakProxy
from functools import partial
from itertools import islice
def export_to_png(self, filename, *args, **kwargs):
    """Saves an image of the widget and its children in png format at the
        specified filename. Works by removing the widget canvas from its
        parent, rendering to an :class:`~kivy.graphics.fbo.Fbo`, and calling
        :meth:`~kivy.graphics.texture.Texture.save`.

        .. note::

            The image includes only this widget and its children. If you want
            to include widgets elsewhere in the tree, you must call
            :meth:`~Widget.export_to_png` from their common parent, or use
            :meth:`~kivy.core.window.WindowBase.screenshot` to capture the
            whole window.

        .. note::

            The image will be saved in png format, you should include the
            extension in your filename.

        .. versionadded:: 1.9.0

        :Parameters:
            `filename`: str
                The filename with which to save the png.
            `scale`: float
                The amount by which to scale the saved image, defaults to 1.

                .. versionadded:: 1.11.0
        """
    self.export_as_image(*args, **kwargs).save(filename, flipped=False)