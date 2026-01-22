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
def handle_dict_items(self, object, name, old, new):
    """ Handles a trait change for items of a dictionary trait.
        """
    self.handle_dict(object, name, new.removed, new.added)
    if len(new.changed) > 0:
        if name.endswith('_items'):
            name = name[:-len('_items')]
        dict = getattr(object, name)
        unregister = self.next.unregister
        register = self.next.register
        for key, obj in new.changed.items():
            unregister(obj)
            register(dict[key])