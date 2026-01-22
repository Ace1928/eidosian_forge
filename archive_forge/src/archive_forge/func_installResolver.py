from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
def installResolver(resolver: IResolverSimple) -> IResolverSimple:
    """
        Set the internal resolver to use to for name lookups.

        @param resolver: The new resolver to use.

        @return: The previously installed resolver.
        """