import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def consumes(self, method, params):
    """Check whether the given parameters are consumed by the given method.

        .. versionadded:: 1.4

        Parameters
        ----------
        method : str
            The name of the method to check.

        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by the given method.
        """
    res = set()
    if self._self_request:
        res = res | self._self_request.consumes(method=method, params=params)
    for _, route_mapping in self._route_mappings.items():
        for callee, caller in route_mapping.mapping:
            if caller == method:
                res = res | route_mapping.router.consumes(method=callee, params=params)
    return res