import pandas as pd
import pytest
from panel.pane import Perspective
from panel.tests.util import serve_component, wait_until
@pytest.fixture
def perspective_data():
    data = {'A': [0.0, 1.0, 2.0, 3.0, 4.0], 'B': [0.0, 1.0, 0.0, 1.0, 0.0], 'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'D': pd.bdate_range('1/1/2009', periods=5)}
    return pd.DataFrame(data)