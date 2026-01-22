def uninstall_twisted_reactor():
    """Uninstalls the Kivy's threaded Twisted Reactor. No more Twisted
    tasks will run after this got called. Use this to clean the
    `twisted.internet.reactor` .

    .. versionadded:: 1.9.0
    """
    import twisted
    if not hasattr(twisted, '_kivy_twisted_reactor_installed'):
        return
    from kivy.base import EventLoop
    global _twisted_reactor_stopper
    _twisted_reactor_stopper()
    EventLoop.unbind(on_stop=_twisted_reactor_stopper)
    del twisted._kivy_twisted_reactor_installed