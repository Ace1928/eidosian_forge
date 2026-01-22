import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def populate_menu(self, menu):
    """
        Populate the provided menu with spelling items.

        :param menu: The menu to populate.
        """
    if not _IS_GTK3:
        menu.remove_all()
    if not self._enabled:
        return
    if _IS_GTK3:
        separator = Gtk.SeparatorMenuItem.new()
        separator.show()
        menu.prepend(separator)
        languages = Gtk.MenuItem.new_with_label(_('Languages'))
        languages.set_submenu(self._get_languages_menu())
        languages.show_all()
        menu.prepend(languages)
    else:
        menu.append_item(self._get_languages_menu())
    if self._marks['click'].inside_word:
        start, end = self._marks['click'].word
        if start.has_tag(self._misspelled):
            word = self._buffer.get_text(start, end, False)
            items = self._suggestion_menu(word)
            if self.collapse:
                menu_label = _('Suggestions')
                if _IS_GTK3:
                    suggestions = Gtk.MenuItem.new_with_label(menu_label)
                    submenu = Gtk.Menu.new()
                else:
                    suggestions = Gio.MenuItem.new(menu_label, None)
                    submenu = Gio.Menu.new()
                for item in items:
                    if _IS_GTK3:
                        submenu.append(item)
                    else:
                        submenu.append_item(item)
                suggestions.set_submenu(submenu)
                if _IS_GTK3:
                    suggestions.show_all()
                    menu.prepend(suggestions)
                else:
                    menu.prepend_item(suggestions)
            else:
                items.reverse()
                for item in items:
                    if _IS_GTK3:
                        menu.prepend(item)
                        menu.show_all()
                    else:
                        menu.prepend_item(item)