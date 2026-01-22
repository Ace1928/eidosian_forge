from . import processMagmaFile
from . import processRurFile
from . import processComponents
def parse_solutions(text, numerical=False):
    """
    Reads the text containing the solutions from a magma computation
    or a rur computation and returns a list of solutions.

    A non-zero dimensional component of the variety is reported as
    NonZeroDimensionalComponent.
    """
    return parse_decomposition(text).solutions(numerical)