from . import commit, controldir, errors, revision
def start_series(self):
    """We will be creating a series of commits.

        This allows us to hold open the locks while we are processing.

        Make sure to call 'finish_series' when you are done.
        """
    if self._tree is not None:
        raise AssertionError('You cannot start a new series while a series is already going.')
    self._tree = self._branch.create_memorytree()
    self._tree.lock_write()