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
@inference_state_method_cache()
def create_instance_context(self, class_context, node):
    new = node
    while True:
        func_node = new
        new = search_ancestor(new, 'funcdef', 'classdef')
        if class_context.tree_node is new:
            func = FunctionValue.from_context(class_context, func_node)
            bound_method = BoundMethod(self, class_context, func)
            if func_node.name.value == '__init__':
                context = bound_method.as_context(self._arguments)
            else:
                context = bound_method.as_context()
            break
    return context.create_context(node)