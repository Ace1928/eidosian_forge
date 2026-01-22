import difflib
from pathlib import Path
from typing import Dict, Iterable, Tuple
from parso import split_lines
from jedi.api.exceptions import RefactoringError
from jedi.inference.value.namespace import ImplicitNSName
def get_new_code(self):
    return self._inference_state.grammar.refactor(self._module_node, self._node_to_str_map)