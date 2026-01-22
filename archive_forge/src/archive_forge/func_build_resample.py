from .default import DefaultMethod
@classmethod
def build_resample(cls, func, squeeze_self):
    """
        Build function that resamples time-series data and does aggregation.

        Parameters
        ----------
        func : callable
            Aggregation function to execute under resampled frame.
        squeeze_self : bool
            Whether or not to squeeze frame before resampling.

        Returns
        -------
        callable
            Function that takes pandas DataFrame and applies aggregation
            to resampled time-series data.
        """

    def fn(df, resample_kwargs, *args, **kwargs):
        """Resample time-series data of the passed frame and apply specified aggregation."""
        if squeeze_self:
            df = df.squeeze(axis=1)
        resampler = df.resample(**resample_kwargs)
        if type(func) is property:
            return func.fget(resampler)
        return func(resampler, *args, **kwargs)
    return fn