import re
import warnings
from collections import defaultdict
from dataclasses import is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
from uuid import UUID
from typing_extensions import Annotated, Literal
from .fields import (
from .json import pydantic_encoder
from .networks import AnyUrl, EmailStr
from .types import (
from .typing import (
from .utils import ROOT_KEY, get_model, lenient_issubclass
def get_annotation_with_constraints(annotation: Any, field_info: FieldInfo) -> Tuple[Type[Any], Set[str]]:
    """
    Get an annotation with used constraints implemented for numbers and strings based on the field_info.

    :param annotation: an annotation from a field specification, as ``str``, ``ConstrainedStr``
    :param field_info: an instance of FieldInfo, possibly with declarations for validations and JSON Schema
    :return: the same ``annotation`` if unmodified or a new annotation along with the used constraints.
    """
    used_constraints: Set[str] = set()

    def go(type_: Any) -> Type[Any]:
        if is_literal_type(type_) or isinstance(type_, ForwardRef) or lenient_issubclass(type_, (ConstrainedList, ConstrainedSet, ConstrainedFrozenSet)):
            return type_
        origin = get_origin(type_)
        if origin is not None:
            args: Tuple[Any, ...] = get_args(type_)
            if any((isinstance(a, ForwardRef) for a in args)):
                return type_
            if origin is Annotated:
                return go(args[0])
            if is_union(origin):
                return Union[tuple((go(a) for a in args))]
            if issubclass(origin, List) and (field_info.min_items is not None or field_info.max_items is not None or field_info.unique_items is not None):
                used_constraints.update({'min_items', 'max_items', 'unique_items'})
                return conlist(go(args[0]), min_items=field_info.min_items, max_items=field_info.max_items, unique_items=field_info.unique_items)
            if issubclass(origin, Set) and (field_info.min_items is not None or field_info.max_items is not None):
                used_constraints.update({'min_items', 'max_items'})
                return conset(go(args[0]), min_items=field_info.min_items, max_items=field_info.max_items)
            if issubclass(origin, FrozenSet) and (field_info.min_items is not None or field_info.max_items is not None):
                used_constraints.update({'min_items', 'max_items'})
                return confrozenset(go(args[0]), min_items=field_info.min_items, max_items=field_info.max_items)
            for t in (Tuple, List, Set, FrozenSet, Sequence):
                if issubclass(origin, t):
                    return t[tuple((go(a) for a in args))]
            if issubclass(origin, Dict):
                return Dict[args[0], go(args[1])]
        attrs: Optional[Tuple[str, ...]] = None
        constraint_func: Optional[Callable[..., type]] = None
        if isinstance(type_, type):
            if issubclass(type_, (SecretStr, SecretBytes)):
                attrs = ('max_length', 'min_length')

                def constraint_func(**kw: Any) -> Type[Any]:
                    return type(type_.__name__, (type_,), kw)
            elif issubclass(type_, str) and (not issubclass(type_, (EmailStr, AnyUrl))):
                attrs = ('max_length', 'min_length', 'regex')
                if issubclass(type_, StrictStr):

                    def constraint_func(**kw: Any) -> Type[Any]:
                        return type(type_.__name__, (type_,), kw)
                else:
                    constraint_func = constr
            elif issubclass(type_, bytes):
                attrs = ('max_length', 'min_length', 'regex')
                if issubclass(type_, StrictBytes):

                    def constraint_func(**kw: Any) -> Type[Any]:
                        return type(type_.__name__, (type_,), kw)
                else:
                    constraint_func = conbytes
            elif issubclass(type_, numeric_types) and (not issubclass(type_, (ConstrainedInt, ConstrainedFloat, ConstrainedDecimal, ConstrainedList, ConstrainedSet, ConstrainedFrozenSet, bool))):
                attrs = ('gt', 'lt', 'ge', 'le', 'multiple_of')
                if issubclass(type_, float):
                    attrs += ('allow_inf_nan',)
                if issubclass(type_, Decimal):
                    attrs += ('max_digits', 'decimal_places')
                numeric_type = next((t for t in numeric_types if issubclass(type_, t)))
                constraint_func = _map_types_constraint[numeric_type]
        if attrs:
            used_constraints.update(set(attrs))
            kwargs = {attr_name: attr for attr_name, attr in ((attr_name, getattr(field_info, attr_name)) for attr_name in attrs) if attr is not None}
            if kwargs:
                constraint_func = cast(Callable[..., type], constraint_func)
                return constraint_func(**kwargs)
        return type_
    return (go(annotation), used_constraints)