import warnings
def be_quiet(self, state):
    """Tell the UI to be more quiet, or not.

        Typically this suppresses progress bars; the application may also look
        at ui_factory.is_quiet().
        """
    self._quiet = state