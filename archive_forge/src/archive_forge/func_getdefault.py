importing any other Kivy modules. Ideally, this means setting them right at
from collections import OrderedDict
from os import environ
from os.path import exists
from weakref import ref
from kivy import kivy_config_fn
from kivy.compat import PY2, string_types
from kivy.logger import Logger, logger_config_update
from kivy.utils import platform
def getdefault(self, section, option, defaultvalue):
    """Get the value of an option in the specified section. If not found,
        it will return the default value.
        """
    if not self.has_section(section):
        return defaultvalue
    if not self.has_option(section, option):
        return defaultvalue
    return self.get(section, option)