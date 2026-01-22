from abc import abstractproperty
from parso.tree import search_ancestor
from jedi import debug
from jedi import settings
from jedi.inference import compiled
from jedi.inference.compiled.value import CompiledValueFilter
from jedi.inference.helpers import values_from_qualified_names, is_big_annoying_library
from jedi.inference.filters import AbstractFilter, AnonymousFunctionExecutionFilter
from jedi.inference.names import ValueName, TreeNameDefinition, ParamName, \
from jedi.inference.base_value import Value, NO_VALUES, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import ValuesArguments, TreeArgumentsWrapper
from jedi.inference.value.function import \
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.dynamic_arrays import get_dynamic_array_instance
from jedi.parser_utils import function_is_staticmethod, function_is_classmethod
def py__getattribute__alternatives(self, string_name):
    """
        Since nothing was inferred, now check the __getattr__ and
        __getattribute__ methods. Stubs don't need to be checked, because
        they don't contain any logic.
        """
    if self.is_stub():
        return NO_VALUES
    name = compiled.create_simple_object(self.inference_state, string_name)
    if is_big_annoying_library(self.parent_context):
        return NO_VALUES
    names = self.get_function_slot_names('__getattr__') or self.get_function_slot_names('__getattribute__')
    return self.execute_function_slots(names, name)