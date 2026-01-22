import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def get_routing_for_object(obj=None):
    """Get a ``Metadata{Router, Request}`` instance from the given object.

    This function returns a
    :class:`~sklearn.utils.metadata_routing.MetadataRouter` or a
    :class:`~sklearn.utils.metadata_routing.MetadataRequest` from the given input.

    This function always returns a copy or an instance constructed from the
    input, such that changing the output of this function will not change the
    original object.

    .. versionadded:: 1.3

    Parameters
    ----------
    obj : object
        - If the object is already a
            :class:`~sklearn.utils.metadata_routing.MetadataRequest` or a
            :class:`~sklearn.utils.metadata_routing.MetadataRouter`, return a copy
            of that.
        - If the object provides a `get_metadata_routing` method, return a copy
            of the output of that method.
        - Returns an empty :class:`~sklearn.utils.metadata_routing.MetadataRequest`
            otherwise.

    Returns
    -------
    obj : MetadataRequest or MetadataRouting
        A ``MetadataRequest`` or a ``MetadataRouting`` taken or created from
        the given object.
    """
    if hasattr(obj, 'get_metadata_routing'):
        return deepcopy(obj.get_metadata_routing())
    elif getattr(obj, '_type', None) in ['metadata_request', 'metadata_router']:
        return deepcopy(obj)
    return MetadataRequest(owner=None)