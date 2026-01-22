import abc
import copy as copy_module
import inspect
import os
import pickle
import re
import types
import warnings
import weakref
from types import FunctionType
from . import __version__ as TraitsVersion
from .adaptation.adaptation_error import AdaptationError
from .constants import DefaultValue, TraitKind
from .ctrait import CTrait, __newobj__
from .ctraits import CHasTraits
from .observation import api as observe_api
from .traits import (
from .trait_types import Any, Bool, Disallow, Event, Python, Str
from .trait_notifiers import (
from .trait_base import (
from .trait_errors import TraitError
from .util.deprecated import deprecated
from .util._traitsui_helpers import check_traitsui_major_version
from .trait_converters import check_trait, mapped_trait_for, trait_for
def sync_trait(self, trait_name, object, alias=None, mutual=True, remove=False):
    """Synchronizes the value of a trait attribute on this object with a
        trait attribute on another object.

        In mutual synchronization, any change to the value of the specified
        trait attribute of either object results in the same value being
        assigned to the corresponding trait attribute of the other object.
        In one-way synchronization, any change to the value of the attribute
        on this object causes the corresponding trait attribute of *object* to
        be updated, but not vice versa.

        For ``List`` traits, the list's items are also synchronized, so that
        mutations to this trait's list will be reflected in the synchronized
        list (and vice versa in the case of mutual synchronization). For
        ``Dict`` and ``Set`` traits, items are not synchronized.

        Parameters
        ----------
        name : str
            Name of the trait attribute on this object.
        object : object
            The object with which to synchronize.
        alias : str
            Name of the trait attribute on *other*; if None or omitted, same
            as *name*.
        mutual : bool or int
            Indicates whether synchronization is mutual (True or non-zero)
            or one-way (False or zero)
        remove : bool or int
            Indicates whether synchronization is being added (False or zero)
            or removed (True or non-zero)
        """
    if alias is None:
        alias = trait_name
    is_list = self._is_list_trait(trait_name) and object._is_list_trait(alias)
    if remove:
        info = self._get_sync_trait_info()
        dic = info.get(trait_name)
        if dic is not None:
            key = (id(object), alias)
            if key in dic:
                del dic[key]
                if len(dic) == 0:
                    del info[trait_name]
                    self._on_trait_change(self._sync_trait_modified, trait_name, remove=True)
                    if is_list:
                        self._on_trait_change(self._sync_trait_items_modified, trait_name + '_items', remove=True)
        if mutual:
            object.sync_trait(alias, self, trait_name, False, True)
        return

    def _sync_trait_listener_deleted(ref, info):
        for key, dic in list(info.items()):
            if key != '':
                for name, value in list(dic.items()):
                    if ref is value[0]:
                        del dic[name]
                if len(dic) == 0:
                    del info[key]
    info = self._get_sync_trait_info()
    dic = info.setdefault(trait_name, {})
    key = (id(object), alias)
    callback = lambda ref: _sync_trait_listener_deleted(ref, info)
    value = (weakref.ref(object, callback), alias)
    if key not in dic:
        if len(dic) == 0:
            self._on_trait_change(self._sync_trait_modified, trait_name)
            if is_list:
                self._on_trait_change(self._sync_trait_items_modified, trait_name + '_items')
        dic[key] = value
        setattr(object, alias, getattr(self, trait_name))
    if mutual:
        object.sync_trait(alias, self, trait_name, False)