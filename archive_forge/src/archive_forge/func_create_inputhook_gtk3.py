from gi.repository import Gtk, GLib  # @UnresolvedImport
def create_inputhook_gtk3(stdin_file):

    def inputhook_gtk3():
        GLib.io_add_watch(stdin_file, GLib.IO_IN, _main_quit)
        Gtk.main()
        return 0
    return inputhook_gtk3