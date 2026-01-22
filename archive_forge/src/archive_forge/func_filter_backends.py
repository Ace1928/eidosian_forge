from __future__ import annotations
import logging
from collections.abc import Callable
from qiskit.providers.backend import Backend
def filter_backends(backends: list[Backend], filters: Callable=None, **kwargs) -> list[Backend]:
    """Return the backends matching the specified filtering.

    Filter the `backends` list by their `configuration` or `status`
    attributes, or from a boolean callable. The criteria for filtering can
    be specified via `**kwargs` or as a callable via `filters`, and the
    backends must fulfill all specified conditions.

    Args:
        backends (list[Backend]): list of backends.
        filters (callable): filtering conditions as a callable.
        **kwargs: dict of criteria.

    Returns:
        list[Backend]: a list of backend instances matching the
            conditions.
    """

    def _match_all(obj, criteria):
        """Return True if all items in criteria matches items in obj."""
        return all((getattr(obj, key_, None) == value_ for key_, value_ in criteria.items()))
    configuration_filters = {}
    status_filters = {}
    for key, value in kwargs.items():
        if all((key in backend.configuration() for backend in backends)):
            configuration_filters[key] = value
        else:
            status_filters[key] = value
    if configuration_filters:
        backends = [b for b in backends if _match_all(b.configuration(), configuration_filters)]
    if status_filters:
        backends = [b for b in backends if _match_all(b.status(), status_filters)]
    backends = list(filter(filters, backends))
    return backends