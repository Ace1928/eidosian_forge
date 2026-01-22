import numpy as np
import pytest
from statsmodels.datasets import anes96
from statsmodels.graphics.boxplots import beanplot, violinplot
@pytest.mark.matplotlib
def test_violinplot_bw_factor(age_and_labels, close_figures):
    age, labels = age_and_labels
    fig, ax = plt.subplots(1, 1)
    violinplot(age, ax=ax, labels=labels, plot_opts={'cutoff_val': 5, 'cutoff_type': 'abs', 'label_fontsize': 'small', 'label_rotation': 30, 'bw_factor': 0.2})