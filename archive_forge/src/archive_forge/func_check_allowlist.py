from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Sequence
from tornado import netutil
def check_allowlist(host: str, allowlist: Sequence[str]) -> bool:
    """ Check a given request host against a allowlist.

    Args:
        host (str) :
            A host string to compare against a allowlist.

            If the host does not specify a port, then ``":80"`` is implicitly
            assumed.

        allowlist (seq[str]) :
            A list of host patterns to match against

    Returns:
        ``True``, if ``host`` matches any pattern in ``allowlist``, otherwise
        ``False``

     """
    if ':' not in host:
        host = host + ':80'
    if host in allowlist:
        return True
    return any((match_host(host, pattern) for pattern in allowlist))