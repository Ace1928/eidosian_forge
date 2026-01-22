import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def process_env_var(env_var: str) -> Variable:
    if env_var == 'platform_python_implementation' or env_var == 'python_implementation':
        return Variable('platform_python_implementation')
    else:
        return Variable(env_var)