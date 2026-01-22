import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
Test a node using the SignalExtraction interface.
        Unlike interface.run(), node.run() checks the traits
        