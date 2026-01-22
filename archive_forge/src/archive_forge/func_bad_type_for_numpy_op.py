import warnings
from typing import NoReturn, Set
from modin.logging import get_logger
from modin.utils import get_current_execution
@classmethod
def bad_type_for_numpy_op(cls, function_name: str, operand_type: type) -> None:
    cls.single_warning(f'Modin NumPy only supports objects of modin.numpy.array types for {function_name}, not {operand_type}. Defaulting to NumPy.')