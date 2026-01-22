import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
def fig8_points():
    pts = [(1564, 148, 0, 1117), (765, 1137, 786, 1117), (1117, 1882, 1490, 1117), (1469, 1137, 786, 1117), (698, 280, 0, 1117), (1166, 372, -744, 1117), (1862, 372, -1440, 1117), (1514, 1068, -744, 1117), (380, 198, 0, 279), (564, 604, 604, 559), (701, 219, 438, 1117), (-679, -241, 0, 1117), (460, 679, -482, 1117)]
    return [[Vector3([a, b, c]) / d for a, b, c, d in pts]]