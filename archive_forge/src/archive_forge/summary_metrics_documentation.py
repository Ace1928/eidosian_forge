from warnings import simplefilter
import numpy as np
import sklearn
import wandb
from wandb.sklearn import utils
Calculate summary metrics for both regressors and classifiers.

    Called by plot_summary_metrics to visualize metrics. Please use the function
    plot_summary_metrics() if you wish to visualize your summary metrics.
    