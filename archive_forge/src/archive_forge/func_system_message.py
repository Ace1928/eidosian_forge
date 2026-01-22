from distutils.core import Command
from distutils.errors import DistutilsSetupError
def system_message(self, level, message, *children, **kwargs):
    self.messages.append((level, message, children, kwargs))
    return nodes.system_message(message, *children, level=level, type=self.levels[level], **kwargs)