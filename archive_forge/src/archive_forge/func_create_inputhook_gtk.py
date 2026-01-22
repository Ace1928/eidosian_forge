import gtk, gobject  # @UnresolvedImport
def create_inputhook_gtk(stdin_file):

    def inputhook_gtk():
        gobject.io_add_watch(stdin_file, gobject.IO_IN, _main_quit)
        gtk.main()
        return 0
    return inputhook_gtk