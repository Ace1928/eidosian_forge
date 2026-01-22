import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
def mean_map(dfgb, **kwargs):
    return pandas.concat([dfgb.sum(**kwargs), dfgb.count()], axis=1, copy=False, keys=['sum', 'count'], names=[GroupByReduce.ID_LEVEL_NAME])