import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def new_text():
    combo = Gtk.ComboBox()
    model = Gtk.ListStore(str)
    combo.set_model(model)
    combo.set_entry_text_column(0)
    return combo