from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_container_for_node(node_id: NodeId, container_name: typing.Optional[str]=None, physical_axes: typing.Optional[PhysicalAxes]=None, logical_axes: typing.Optional[LogicalAxes]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Optional[NodeId]]:
    """
    Returns the query container of the given node based on container query
    conditions: containerName, physical, and logical axes. If no axes are
    provided, the style container is returned, which is the direct parent or the
    closest element with a matching container-name.

    **EXPERIMENTAL**

    :param node_id:
    :param container_name: *(Optional)*
    :param physical_axes: *(Optional)*
    :param logical_axes: *(Optional)*
    :returns: *(Optional)* The container node for the given node, or null if not found.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    if container_name is not None:
        params['containerName'] = container_name
    if physical_axes is not None:
        params['physicalAxes'] = physical_axes.to_json()
    if logical_axes is not None:
        params['logicalAxes'] = logical_axes.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getContainerForNode', 'params': params}
    json = (yield cmd_dict)
    return NodeId.from_json(json['nodeId']) if 'nodeId' in json else None