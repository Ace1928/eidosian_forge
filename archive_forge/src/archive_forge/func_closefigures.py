import os
import matplotlib.pyplot as plt
import pytest
@pytest.fixture
def closefigures():
    yield
    plt.close('all')