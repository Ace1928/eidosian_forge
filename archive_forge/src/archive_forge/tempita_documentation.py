import argparse
import os
from Cython import Tempita as tempita
Process tempita templated file and write out the result.

    The template file is expected to end in `.c.tp` or `.pyx.tp`:
    E.g. processing `template.c.in` generates `template.c`.

    