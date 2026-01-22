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
def register_for_motion_event(self, type_id, widget=None):
    """Register to receive motion events of `type_id`.

        Override :meth:`on_motion` or bind to `on_motion` event to handle
        the incoming motion events.

        :Parameters:
            `type_id`: `str`
                Motion event type id (eg. "touch", "hover", etc.)
            `widget`: `Widget`
                Child widget or `self` if omitted

        .. versionadded:: 2.1.0

        .. note::
            Method can be called multiple times with the same arguments.

        .. warning::
            This is an experimental method and it remains so while this warning
            is present.
        """
    a_widget = widget or self
    motion_filter = self.motion_filter
    if type_id not in motion_filter:
        motion_filter[type_id] = [a_widget]
    elif widget not in motion_filter[type_id]:
        index = self._find_index_in_motion_filter(type_id, a_widget)
        motion_filter[type_id].insert(index, a_widget)