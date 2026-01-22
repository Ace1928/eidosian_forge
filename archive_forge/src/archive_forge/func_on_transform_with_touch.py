from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def on_transform_with_touch(self, touch):
    """
        Called when a touch event has transformed the scatter widget.
        By default this does nothing, but can be overridden by derived
        classes that need to react to transformations caused by user
        input.

        :Parameters:
            `touch`:
                The touch object which triggered the transformation.

        .. versionadded:: 1.8.0
        """
    pass