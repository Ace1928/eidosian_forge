from IPython.core.getipython import get_ipython
def is_event_loop_running_wx(app=None):
    """Is the wx event loop running."""
    ip = get_ipython()
    if ip is not None:
        if ip.active_eventloop and ip.active_eventloop == 'wx':
            return True
    if app is None:
        app = get_app_wx()
    if hasattr(app, '_in_event_loop'):
        return app._in_event_loop
    else:
        return app.IsMainLoopRunning()