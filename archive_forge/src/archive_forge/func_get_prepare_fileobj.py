from __future__ import annotations
import io
import typing as ty
from copy import copy
from .openers import ImageOpener
def get_prepare_fileobj(self, *args, **kwargs) -> ImageOpener:
    """Return fileobj if present, or return fileobj from filename

        Set position to that given in self.pos

        Parameters
        ----------
        *args : tuple
           positional arguments to file open.  Ignored if there is a
           defined ``self.fileobj``.  These might include the mode, such
           as 'rb'
        **kwargs : dict
           named arguments to file open.  Ignored if there is a
           defined ``self.fileobj``

        Returns
        -------
        fileobj : file-like object
           object has position set (via ``fileobj.seek()``) to
           ``self.pos``
        """
    if self.fileobj is not None:
        obj = ImageOpener(self.fileobj)
        obj.seek(self.pos)
    elif self.filename is not None:
        obj = ImageOpener(self.filename, *args, **kwargs)
        if self.pos != 0:
            obj.seek(self.pos)
    else:
        raise FileHolderError('No filename or fileobj present')
    return obj