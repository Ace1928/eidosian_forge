from pathlib import Path
from typing import Optional
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.filters import DictFilter
from jedi.inference.names import ValueNameMixin, AbstractNameDefinition
from jedi.inference.base_value import Value
from jedi.inference.value.module import SubModuleDictMixin
from jedi.inference.context import NamespaceContext
def py__package__(self):
    """Return the fullname
        """
    return self.string_names