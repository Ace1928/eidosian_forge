import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
@property
def most_specific(self) -> Subdivision:
    """The most specific (smallest) subdivision available.

        If there are no :py:class:`Subdivision` objects for the response,
        this returns an empty :py:class:`Subdivision`.

        :type: :py:class:`Subdivision`
        """
    try:
        return self[-1]
    except IndexError:
        return Subdivision(self._locales)