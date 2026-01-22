import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
Reverse all effects of the enable_xxx() calls except for
    require_version() calls and imports.
    