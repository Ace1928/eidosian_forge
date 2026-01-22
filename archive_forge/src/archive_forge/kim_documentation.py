from . import kimpy_wrappers
from .exceptions import KIMCalculatorError
from .calculators import (

    Returns True if the model specified is a KIM Portable Model (if it
    is not, then it must be a KIM Simulator Model -- there are no other
    types of models in KIM)
    