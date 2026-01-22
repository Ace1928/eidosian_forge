from . import errors, osutils
def iter_all(self):
    """Match up the values in the different trees."""
    yield from self._walk_master_tree()
    self._finish_others()
    yield from self._walk_others()