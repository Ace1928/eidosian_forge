from pathlib import Path
import numpy as np
import pytest
from ase.utils.xrdebye import XrDebye, wavelengths
from ase.cluster.cubic import FaceCenteredCubic
Tests for XrDebye class