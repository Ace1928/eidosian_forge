import numpy as np
import pytest
from statsmodels.datasets import anes96
from statsmodels.graphics.boxplots import beanplot, violinplot
@pytest.mark.matplotlib
def test_beanplot_side_right(age_and_labels, close_figures):
    age, labels = age_and_labels
    fig, ax = plt.subplots(1, 1)
    beanplot(age, ax=ax, labels=labels, jitter=True, side='right', plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs', 'label_fontsize': 'small', 'label_rotation': 30})