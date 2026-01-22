from __future__ import annotations
import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote
from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry
def render_as_string(self, hide_password: bool=True) -> str:
    """Render this :class:`_engine.URL` object as a string.

        This method is used when the ``__str__()`` or ``__repr__()``
        methods are used.   The method directly includes additional options.

        :param hide_password: Defaults to True.   The password is not shown
         in the string unless this is set to False.

        """
    s = self.drivername + '://'
    if self.username is not None:
        s += quote(self.username, safe=' +')
        if self.password is not None:
            s += ':' + ('***' if hide_password else quote(str(self.password), safe=' +'))
        s += '@'
    if self.host is not None:
        if ':' in self.host:
            s += f'[{self.host}]'
        else:
            s += self.host
    if self.port is not None:
        s += ':' + str(self.port)
    if self.database is not None:
        s += '/' + self.database
    if self.query:
        keys = list(self.query)
        keys.sort()
        s += '?' + '&'.join((f'{quote_plus(k)}={quote_plus(element)}' for k in keys for element in util.to_list(self.query[k])))
    return s