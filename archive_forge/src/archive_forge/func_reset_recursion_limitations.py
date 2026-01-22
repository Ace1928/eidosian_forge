import parso
from jedi.file_io import FileIO
from jedi import debug
from jedi import settings
from jedi.inference import imports
from jedi.inference import recursion
from jedi.inference.cache import inference_state_function_cache
from jedi.inference import helpers
from jedi.inference.names import TreeNameDefinition
from jedi.inference.base_value import ContextualizedNode, \
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.syntax_tree import infer_expr_stmt, \
from jedi.inference.imports import follow_error_node_imports_if_possible
from jedi.plugins import plugin_manager
def reset_recursion_limitations(self):
    self.recursion_detector = recursion.RecursionDetector()
    self.execution_recursion_detector = recursion.ExecutionRecursionDetector(self)