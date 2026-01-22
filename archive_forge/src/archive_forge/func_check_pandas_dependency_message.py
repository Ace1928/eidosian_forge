import inspect
import os
import numpy as np
import pytest
import sklearn.datasets
def check_pandas_dependency_message(fetch_func):
    try:
        import pandas
        pytest.skip('This test requires pandas to not be installed')
    except ImportError:
        name = fetch_func.__name__
        expected_msg = f'{name} with as_frame=True requires pandas'
        with pytest.raises(ImportError, match=expected_msg):
            fetch_func(as_frame=True)