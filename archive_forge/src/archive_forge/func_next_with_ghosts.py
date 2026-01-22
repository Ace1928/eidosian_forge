import time
from . import debug, errors, osutils, revision, trace
def next_with_ghosts(self):
    """Return the next found ancestors, with ghosts split out.

        Ancestors are returned in the order they are seen in a breadth-first
        traversal.  No ancestor will be returned more than once. Ancestors are
        returned only after asking for their parents, which allows us to detect
        which revisions are ghosts and which are not.

        :return: A tuple with (present ancestors, ghost ancestors) sets.
        """
    if self._returning != 'next_with_ghosts':
        self._returning = 'next_with_ghosts'
        self._advance()
    if len(self._next_query) == 0:
        raise StopIteration()
    self._advance()
    return (self._current_present, self._current_ghosts)