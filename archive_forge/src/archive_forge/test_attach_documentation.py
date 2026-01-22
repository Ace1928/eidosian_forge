import pytest
import numpy as np
from ase.parallel import world
from ase.build import molecule, fcc111
from ase.build.attach import (attach, attach_randomly,
Check that every core has its own structure