from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import ListProperty, ObjectProperty, BooleanProperty
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
Each element in a dropdown list uses a default/user-supplied height.
    Set to True to propagate the Spinner's height value to each dropdown
    list element.

    .. versionadded:: 1.10.0

    :attr:`sync_height` is a :class:`~kivy.properties.BooleanProperty` and
    defaults to False.
    