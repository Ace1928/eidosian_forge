from _dbus_glib_bindings import DBusGMainLoop, gthreads_init
Initialize threads in dbus-glib, if this has not already been done.

    This must be called before creating a second thread in a program that
    uses this module.
    