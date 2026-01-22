from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
def register_event_manager(self, manager):
    """Register and start an event manager to handle events declared in
        :attr:`~kivy.eventmanager.EventManagerBase.type_ids` attribute.

        .. versionadded:: 2.1.0

        .. warning::
            This is an experimental method and it remains so until this warning
            is present as it can be changed or removed in the next versions of
            Kivy.
        """
    self.event_managers.insert(0, manager)
    for type_id in manager.type_ids:
        self.event_managers_dict[type_id].insert(0, manager)
    manager.window = self
    manager.start()