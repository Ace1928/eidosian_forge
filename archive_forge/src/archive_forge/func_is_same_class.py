from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
def is_same_class(self, other):
    if not isinstance(other, DefineGenericBaseClass):
        return False
    if self.tree_node != other.tree_node:
        return False
    given_params1 = self.get_generics()
    given_params2 = other.get_generics()
    if len(given_params1) != len(given_params2):
        return False
    return all((any((cls2.is_same_class(cls1) for cls1 in class_set1.gather_annotation_classes() for cls2 in class_set2.gather_annotation_classes())) for class_set1, class_set2 in zip(given_params1, given_params2)))