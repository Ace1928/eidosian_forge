from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_properties(object_id: RemoteObjectId, own_properties: typing.Optional[bool]=None, accessor_properties_only: typing.Optional[bool]=None, generate_preview: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.List[PropertyDescriptor], typing.Optional[typing.List[InternalPropertyDescriptor]], typing.Optional[typing.List[PrivatePropertyDescriptor]], typing.Optional[ExceptionDetails]]]:
    """
    Returns properties of a given object. Object group of the result is inherited from the target
    object.

    :param object_id: Identifier of the object to return properties for.
    :param own_properties: *(Optional)* If true, returns properties belonging only to the element itself, not to its prototype chain.
    :param accessor_properties_only: **(EXPERIMENTAL)** *(Optional)* If true, returns accessor properties (with getter/setter) only; internal properties are not returned either.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the results.
    :returns: A tuple with the following items:

        0. **result** - Object properties.
        1. **internalProperties** - *(Optional)* Internal object properties (only of the element itself).
        2. **privateProperties** - *(Optional)* Object private properties.
        3. **exceptionDetails** - *(Optional)* Exception details.
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if own_properties is not None:
        params['ownProperties'] = own_properties
    if accessor_properties_only is not None:
        params['accessorPropertiesOnly'] = accessor_properties_only
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.getProperties', 'params': params}
    json = (yield cmd_dict)
    return ([PropertyDescriptor.from_json(i) for i in json['result']], [InternalPropertyDescriptor.from_json(i) for i in json['internalProperties']] if 'internalProperties' in json else None, [PrivatePropertyDescriptor.from_json(i) for i in json['privateProperties']] if 'privateProperties' in json else None, ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None)