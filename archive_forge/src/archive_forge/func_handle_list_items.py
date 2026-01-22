import re
import string
import weakref
from string import whitespace
from types import MethodType
from .constants import DefaultValue
from .trait_base import Undefined, Uninitialized
from .trait_errors import TraitError
from .trait_notifiers import TraitChangeNotifyWrapper
from .util.weakiddict import WeakIDKeyDict
def handle_list_items(self, object, name, old, new):
    """ Handles a trait change for items of a list (or set) trait.
        """
    self.handle_list(object, name, new.removed, new.added)