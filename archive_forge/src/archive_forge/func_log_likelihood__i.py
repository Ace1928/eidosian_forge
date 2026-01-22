from xarray import apply_ufunc
from ..stats import wrap_xarray_ufunc as _wrap_xarray_ufunc
def log_likelihood__i(self, excluded_obs, idata__i):
    """Get the log likelilhood samples :math:`\\log p_{post(-i)}(y_i)`.

        Calculate the log likelihood of the data contained in excluded_obs using the
        model fitted with this data excluded, the results of which are stored in ``idata__i``.

        Parameters
        ----------
        excluded_obs
            Observations for which to calculate their log likelihood. The second item from
            the tuple returned by `sel_observations` is passed as this argument.
        idata__i: InferenceData
            Inference results of refitting the data excluding some observations. The
            result of `get_inference_data` is used as this argument.

        Returns
        -------
        log_likelihood: xr.Dataarray
            Log likelihood of ``excluded_obs`` evaluated at each of the posterior samples
            stored in ``idata__i``.
        """
    if self.log_lik_fun is None:
        raise NotImplementedError('When `log_like_fun` is not set during class initialization log_likelihood__i method must be overwritten')
    posterior = idata__i.posterior
    arys = (*excluded_obs, *[posterior[var_name] for var_name in self.posterior_vars])
    ufunc_applier = apply_ufunc if self.is_ufunc else _wrap_xarray_ufunc
    log_lik_idx = ufunc_applier(self.log_lik_fun, *arys, kwargs=self.log_lik_kwargs, **self.apply_ufunc_kwargs)
    return log_lik_idx