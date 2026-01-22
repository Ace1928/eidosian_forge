def install_gobject_iteration():
    """Import and install gobject context iteration inside our event loop.
    This is used as soon as gobject is used (like gstreamer).
    """
    from kivy.clock import Clock
    try:
        from gi.repository import GObject as gobject
    except ImportError:
        import gobject
    if hasattr(gobject, '_gobject_already_installed'):
        return
    gobject._gobject_already_installed = True
    loop = gobject.MainLoop()
    gobject.threads_init()
    context = loop.get_context()

    def _gobject_iteration(*largs):
        loop = 0
        while context.pending() and loop < 10:
            context.iteration(False)
            loop += 1
    Clock.schedule_interval(_gobject_iteration, 0)