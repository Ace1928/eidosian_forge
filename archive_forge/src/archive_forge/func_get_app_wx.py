from IPython.core.getipython import get_ipython
def get_app_wx(*args, **kwargs):
    """Create a new wx app or return an exiting one."""
    import wx
    app = wx.GetApp()
    if app is None:
        if 'redirect' not in kwargs:
            kwargs['redirect'] = False
        app = wx.PySimpleApp(*args, **kwargs)
    return app