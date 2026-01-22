import numpy as np
import matplotlib.pyplot as plt
def test_nonstring_label():
    plt.bar(np.arange(10), np.random.rand(10), label=1)
    plt.legend()