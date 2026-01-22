from os import getpid
from typing import Dict, List, Mapping, Optional, Sequence
from attrs import Factory, define
@classmethod
def fromEnvironment(cls, environ: Optional[Mapping[str, str]]=None, start: Optional[int]=None) -> 'ListenFDs':
    """
        @param environ: A dictionary-like object to inspect to discover
            inherited descriptors.  By default, L{None}, indicating that the
            real process environment should be inspected.  The default is
            suitable for typical usage.

        @param start: An integer giving the lowest value of an inherited
            descriptor systemd will give us.  By default, L{None}, indicating
            the known correct (that is, in agreement with systemd) value will be
            used.  The default is suitable for typical usage.

        @return: A new instance of C{cls} which can be used to look up the
            descriptors which have been inherited.
        """
    if environ is None:
        from os import environ as _environ
        environ = _environ
    if start is None:
        start = cls._START
    if str(getpid()) == environ.get('LISTEN_PID'):
        descriptors: List[int] = _parseDescriptors(start, environ)
        names: Sequence[str] = _parseNames(environ)
    else:
        descriptors = []
        names = ()
    if len(names) != len(descriptors):
        return cls([], ())
    return cls(descriptors, names)