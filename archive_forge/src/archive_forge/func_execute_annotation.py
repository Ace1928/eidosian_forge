import re
from functools import partial
from inspect import Parameter
from pathlib import Path
from typing import Optional
from jedi import debug
from jedi.inference.utils import to_list
from jedi.cache import memoize_method
from jedi.inference.filters import AbstractFilter
from jedi.inference.names import AbstractNameDefinition, ValueNameMixin, \
from jedi.inference.base_value import Value, ValueSet, NO_VALUES
from jedi.inference.lazy_value import LazyKnownValue
from jedi.inference.compiled.access import _sentinel
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.helpers import reraise_getitem_errors
from jedi.inference.signature import BuiltinSignature
from jedi.inference.context import CompiledContext, CompiledModuleContext
def execute_annotation(self):
    if self.access_handle.get_repr() == 'None':
        return ValueSet([self])
    name, args = self.access_handle.get_annotation_name_and_args()
    arguments = [ValueSet([create_from_access_path(self.inference_state, path)]) for path in args]
    if name == 'Union':
        return ValueSet.from_sets((arg.execute_annotation() for arg in arguments))
    elif name:
        return ValueSet([v.with_generics(arguments) for v in self.inference_state.typing_module.py__getattribute__(name)]).execute_annotation()
    return super().execute_annotation()