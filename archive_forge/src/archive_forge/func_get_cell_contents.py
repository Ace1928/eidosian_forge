import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def get_cell_contents():
    for name, c in zip(func, closures):
        try:
            cell_contents = c.cell_contents
        except ValueError:
            continue
        yield (name, c.cell_contents)