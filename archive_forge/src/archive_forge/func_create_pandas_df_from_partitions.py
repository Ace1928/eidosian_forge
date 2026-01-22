import pandas
from pandas.api.types import union_categoricals
from modin.error_message import ErrorMessage
def create_pandas_df_from_partitions(partition_data, partition_shape, called_from_remote=False):
    """
    Convert partition data of multiple dataframes to a single dataframe.

    Parameters
    ----------
    partition_data : list
        List of pandas DataFrames or list of Object references holding pandas DataFrames.
    partition_shape : int or tuple
        Shape of the partitions NumPy array.
    called_from_remote : bool, default: False
        Flag used to check if explicit copy should be done in concat.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame.
    """
    if all((isinstance(obj, (pandas.DataFrame, pandas.Series)) for obj in partition_data)):
        height, width, *_ = tuple(partition_shape) + (0,)
        objs = iter(partition_data)
        partition_data = [[next(objs) for _ in range(width)] for __ in range(height)]
    else:
        partition_data = [[obj.to_pandas() for obj in part] for part in partition_data]
    if all((isinstance(part, pandas.Series) for row in partition_data for part in row)):
        axis = 0
    elif all((isinstance(part, pandas.DataFrame) for row in partition_data for part in row)):
        axis = 1
    else:
        ErrorMessage.catch_bugs_and_request_email(True)

    def is_part_empty(part):
        return part.empty and (not isinstance(part, pandas.DataFrame) or len(part.columns) == 0)
    df_rows = [pandas.concat([part for part in row], axis=axis, copy=False) for row in partition_data if not all((is_part_empty(part) for part in row))]
    del partition_data
    if len(df_rows) == 0:
        return pandas.DataFrame()
    else:
        return concatenate(df_rows, copy=not called_from_remote)