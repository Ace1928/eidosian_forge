from collections.abc import Iterable, Sequence
import numpy as np
import pandas as pd
import scipy
import sklearn
import wandb
def warn_chart_limit(limit, chart):
    warning = f'using only the first {limit} datapoints to create chart {chart}'
    wandb.termwarn(warning)