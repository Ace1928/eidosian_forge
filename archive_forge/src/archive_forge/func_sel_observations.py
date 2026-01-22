from xarray import apply_ufunc
from ..stats import wrap_xarray_ufunc as _wrap_xarray_ufunc
def sel_observations(self, idx):
    """Select a subset of the observations in idata_orig.

        **Not implemented**: This method must be implemented by the SamplingWrapper subclasses.
        It is documented here to show its format and call signature.

        Parameters
        ----------
        idx
            Indexes to separate from the rest of the observed data.

        Returns
        -------
        modified_observed_data
            Observed data whose index is *not* ``idx``
        excluded_observed_data
            Observed data whose index is ``idx``
        """
    raise NotImplementedError('sel_observations method must be implemented for each subclass')