from __future__ import annotations
import argparse
import dataclasses
from typing import (
from typing_extensions import Annotated, get_args, get_origin
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def handle_field(field: _fields.FieldDefinition, type_from_typevar: Dict[TypeVar, TypeForm[Any]], parent_classes: Set[Type[Any]], intern_prefix: str, extern_prefix: str, subcommand_prefix: str) -> Union[_arguments.ArgumentDefinition, ParserSpecification, SubparsersSpecification]:
    """Determine what to do with a single field definition."""
    if isinstance(field.type_or_callable, TypeVar):
        raise _instantiators.UnsupportedTypeAnnotationError(f'Field {field.intern_name} has an unbound TypeVar: {field.type_or_callable}.')
    if _markers.Fixed not in field.markers and _markers.Suppress not in field.markers:
        subparsers_attempt = SubparsersSpecification.from_field(field, type_from_typevar=type_from_typevar, parent_classes=parent_classes, intern_prefix=_strings.make_field_name([intern_prefix, field.intern_name]), extern_prefix=_strings.make_field_name([extern_prefix, field.extern_name]))
        if subparsers_attempt is not None:
            if not subparsers_attempt.required and _markers.AvoidSubcommands in field.markers:
                field = dataclasses.replace(field, type_or_callable=type(field.default))
            else:
                return subparsers_attempt
        if _fields.is_nested_type(field.type_or_callable, field.default):
            field = dataclasses.replace(field, type_or_callable=_resolver.unwrap_newtype_and_narrow_subtypes(field.type_or_callable, field.default))
            return ParserSpecification.from_callable_or_type(field.type_or_callable if len(field.markers) == 0 else Annotated.__class_getitem__((field.type_or_callable,) + tuple(field.markers)), description=None, parent_classes=parent_classes, default_instance=field.default, intern_prefix=_strings.make_field_name([intern_prefix, field.intern_name]), extern_prefix=_strings.make_field_name([extern_prefix, field.extern_name]), subcommand_prefix=subcommand_prefix, support_single_arg_types=False)
    return _arguments.ArgumentDefinition(intern_prefix=intern_prefix, extern_prefix=extern_prefix, subcommand_prefix=subcommand_prefix, field=field, type_from_typevar=type_from_typevar)