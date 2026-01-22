from . import schema
from .jsonutil import get_column
from .search import Search
def set_autolearn(self, auto=None, tick=None):
    """ Once in a while queries will persist additional
            information on the server. This information is available
            through the following methods of this class:

                - experiment_types
                - assessor_types
                - scan_types
                - reconstruction_types

            It is also transparently used in insert operations.

            Parameters
            ----------
            auto: boolean
                True to enable auto learn. False to disable.
            tick: int
                Every 'tick' seconds, if a query is issued, additional
                information will be persisted.

            See Also
            --------
            :func:`EObject.insert`
        """
    if auto is not None:
        self._auto = auto
    if tick is not None:
        self._tick = tick