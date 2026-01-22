import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
@staticmethod
def from_foreign_session(session, cls=None):
    """Create a session object matching the foreign `session`.

        Parameters
        ----------
        session : obj
            A foreign session object.
        cls : Session class, optional
            The class to return.

        Returns
        -------
        Session

        """
    if not cls:
        return DummySession()
    else:
        return cls(session)