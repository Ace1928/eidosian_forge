import numpy as np
from ase.build.general_surface import surface
from ase.geometry import get_layers
from ase.symbols import string2symbols

        The above method gives you the boundries of between terminations that
        will allow you to build a complete set of terminations. However, it
        does not return all the boundries. Thus you must check both above and
        below the boundary, and not stray too far from the boundary. If you move
        too far away, you risk hitting another boundary you did not find.
        