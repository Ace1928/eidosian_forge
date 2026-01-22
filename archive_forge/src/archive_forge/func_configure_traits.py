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
def configure_traits(self, filename=None, view=None, kind=None, edit=None, context=None, handler=None, id='', scrollable=None, **args):
    """Creates and displays a dialog box for editing values of trait
        attributes, as if it were a complete, self-contained GUI application.

        This method is intended for use in applications that do not normally
        have a GUI. Control does not resume in the calling application until
        the user closes the dialog box.

        The method attempts to open and unpickle the contents of *filename*
        before displaying the dialog box. When editing is complete, the method
        attempts to pickle the updated contents of the object back to
        *filename*. If the file referenced by *filename* does not exist, the
        object is not modified before displaying the dialog box. If *filename*
        is unspecified or None, no pickling or unpickling occurs.

        If *edit* is True (the default), a dialog box for editing the
        current object is displayed. If *edit* is False or None, no
        dialog box is displayed. You can use ``edit=False`` if you want the
        object to be restored from the contents of *filename*, without being
        modified by the user.

        Parameters
        ----------
        filename : str
            The name (including path) of a file that contains a pickled
            representation of the current object. When this parameter is
            specified, the method reads the corresponding file (if it exists)
            to restore the saved values of the object's traits before
            displaying them. If the user confirms the dialog box (by clicking
            **OK**), the new values are written to the file. If this parameter
            is not specified, the values are loaded from the in-memory object,
            and are not persisted when the dialog box is closed.

            .. deprecated:: 6.0.0

        view : View or str
            A View object (or its name) that defines a user interface for
            editing trait attribute values of the current object. If the view
            is defined as an attribute on this class, use the name of the
            attribute. Otherwise, use a reference to the view object. If this
            attribute is not specified, the View object returned by
            trait_view() is used.
        kind : str
            The type of user interface window to create. See the
            **traitsui.view.kind_trait** trait for values and
            their meanings. If *kind* is unspecified or None, the **kind**
            attribute of the View object is used.
        edit : bool
            Indicates whether to display a user interface. If *filename*
            specifies an existing file, setting *edit* to False loads the
            saved values from that file into the object without requiring
            user interaction.

            .. deprecated:: 6.2.0

        context : object or dictionary
            A single object or a dictionary of string/object pairs, whose trait
            attributes are to be edited. If not specified, the current object
            is used
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
        True on success.
        """
    if filename is not None:
        message = 'Restoring from pickle will not be supported starting with traits 7.0.0'
        warnings.warn(message, DeprecationWarning)
        if os.path.exists(filename):
            with open(filename, 'rb') as fd:
                self.copy_traits(pickle.Unpickler(fd).load())
    if edit is None:
        edit = True
    else:
        message = 'The edit argument to configure_traits is deprecated, and will be removed in Traits 7.0.0'
        warnings.warn(message, DeprecationWarning)
    if edit:
        from traitsui.api import toolkit
        if context is None:
            context = self
        rc = toolkit().view_application(context, self.trait_view(view), kind, handler, id, scrollable, args)
        if rc and filename is not None:
            with open(filename, 'wb') as fd:
                pickle.Pickler(fd, protocol=3).dump(self)
        return rc
    return True