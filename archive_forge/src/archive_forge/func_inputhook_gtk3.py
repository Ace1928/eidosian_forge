from gi.repository import Gtk, GLib  # @UnresolvedImport
def inputhook_gtk3():
    GLib.io_add_watch(stdin_file, GLib.IO_IN, _main_quit)
    Gtk.main()
    return 0