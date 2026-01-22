import inspect
from typing import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from langchain.output_parsers.ernie_functions import (
from langchain.utils.ernie_functions import convert_pydantic_to_ernie_function
def convert_python_function_to_ernie_function(function: Callable) -> Dict[str, Any]:
    """Convert a Python function to an Ernie function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.
    """
    description, arg_descriptions = _parse_python_function_docstring(function)
    return {'name': _get_python_function_name(function), 'description': description, 'parameters': {'type': 'object', 'properties': _get_python_function_arguments(function, arg_descriptions), 'required': _get_python_function_required_args(function)}}