import os
from collections import abc
from functools import partial
from gi.repository import GLib, GObject, Gio
def do_create_closure(self, builder, func_name, flags, obj):
    current_object = builder.get_current_object() or self._scope_object
    if not self._scope_object:
        current_object = builder.get_current_object()
        if func_name not in current_object.__gtktemplate_methods__:
            return None
        current_object.__gtktemplate_handlers__.add(func_name)
        handler_name = current_object.__gtktemplate_methods__[func_name]
    else:
        current_object = self._scope_object
        handler_name = func_name
    swapped = int(flags & Gtk.BuilderClosureFlags.SWAPPED)
    if swapped:
        raise RuntimeError('%r not supported' % GObject.ConnectFlags.SWAPPED)
        return None
    handler, args = _extract_handler_and_args(current_object, handler_name)
    if obj:
        p = partial(handler, *args, swap_data=obj)
    else:
        p = partial(handler, *args)
    p.__gtk_template__ = True
    return p