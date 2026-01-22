import copy
import importlib
from kivy.logger import Logger
from kivy.context import register_context
import kivy.factory_registers  # NOQA
def unregister_from_filename(self, filename):
    """Unregister all the factory objects related to the filename passed in
        the parameter.

        .. versionadded:: 1.7.0
        """
    to_remove = [x for x in self.classes if self.classes[x]['filename'] == filename]
    for name in to_remove:
        del self.classes[name]