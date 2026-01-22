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
def edit_traits(self, view=None, parent=None, kind=None, context=None, handler=None, id='', scrollable=None, **args):
    """ Displays a user interface window for editing trait attribute
        values.

        Parameters
        ----------
        view : View or string
            A View object (or its name) that defines a user interface for
            editing trait attribute values of the current object. If the view
            is defined as an attribute on this class, use the name of the
            attribute. Otherwise, use a reference to the view object. If this
            attribute is not specified, the View object returned by
            trait_view() is used.
        parent : toolkit control
            The reference to a user interface component to use as the parent
            window for the object's UI window.
        kind : str
            The type of user interface window to create. See the
            **traitsui.view.kind_trait** trait for values and
            their meanings. If *kind* is unspecified or None, the **kind**
            attribute of the View object is used.
        context : object or dictionary
            A single object or a dictionary of string/object pairs, whose trait
            attributes are to be edited. If not specified, the current object
            is used.
        handler : Handler
            A handler object used for event handling in the dialog box. If
            None, the default handler for Traits UI is used.
        id : str
            A unique ID for persisting preferences about this user interface,
            such as size and position. If not specified, no user preferences
            are saved.
        scrollable : bool
            Indicates whether the dialog box should be scrollable. When set to
            True, scroll bars appear on the dialog box if it is not large
            enough to display all of the items in the view at one time.

        Returns
        -------
        A UI object.
        """
    if context is None:
        context = self
    view = self.trait_view(view)
    return view.ui(context, parent, kind, self.trait_view_elements(), handler, id, scrollable, args)