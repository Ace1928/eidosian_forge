import os
from pathlib import Path
import pytest
@pytest.mark.parametrize('nb_file', ('examples/01_intro_model_definition_methods.ipynb', 'examples/05_benchmarking_layers.ipynb'))
def test_ipython_notebooks(test_files: None):
    ...