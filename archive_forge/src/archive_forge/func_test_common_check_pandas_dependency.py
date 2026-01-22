import inspect
import os
import numpy as np
import pytest
import sklearn.datasets
@pytest.mark.parametrize('name, dataset_func', _generate_func_supporting_param('as_frame'))
def test_common_check_pandas_dependency(name, dataset_func):
    check_pandas_dependency_message(dataset_func)