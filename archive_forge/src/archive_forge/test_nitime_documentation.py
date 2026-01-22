import os
import tempfile
import numpy as np
import pytest
from nipype.testing import example_data
import nipype.interfaces.nitime as nitime
Test that the coherence analyzer works