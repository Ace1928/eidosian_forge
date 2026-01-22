from . import processMagmaFile
from . import processRurFile
from . import processComponents
def parse_solutions_from_file(filename, numerical=False):
    """
    As parse_solutions, but takes a filename instead.
    """
    with open(filename, 'r') as f:
        return parse_solutions(f.read(), numerical)