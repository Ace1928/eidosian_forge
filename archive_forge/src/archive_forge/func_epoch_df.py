import time
@property
def epoch_df(self):
    """The dataframe with epoch data.
        This has timing information.
        """
    return self._dataframes['epoch']