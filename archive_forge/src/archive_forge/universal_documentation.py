import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
Setting to select smartquote transformations.

    The default 'qDe' educates normal quote characters: (", '),
    em- and en-dashes (---, --) and ellipses (...).
    