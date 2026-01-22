from collections import defaultdict
import numpy as np
from statsmodels.base._penalties import SCADSmoothed
def screen_exog(self, exog, endog=None, maxiter=100, method='bfgs', disp=False, fit_kwds=None):
    """screen and select variables (columns) in exog

        Parameters
        ----------
        exog : ndarray
            candidate explanatory variables that are screened for inclusion in
            the model
        endog : ndarray (optional)
            use a new endog in the screening model.
            This is not tested yet, and might not work correctly
        maxiter : int
            number of screening iterations
        method : str
            optimization method to use in fit, needs to be only of the gradient
            optimizers
        disp : bool
            display option for fit during optimization

        Returns
        -------
        res_screen : instance of ScreeningResults
            The attribute `results_final` contains is the results instance
            with the final model selection.
            `idx_nonzero` contains the index of the selected exog in the full
            exog, combined exog that are always kept plust exog_candidates.
            see ScreeningResults for a full description
        """
    model_class = self.model_class
    if endog is None:
        endog = self.endog
    x0 = self.exog_keep
    k_keep = self.k_keep
    x1 = exog
    k_current = x0.shape[1]
    x = np.column_stack((x0, x1))
    nobs, k_vars = x.shape
    fkwds = fit_kwds if fit_kwds is not None else {}
    fit_kwds = {'maxiter': 200, 'disp': False}
    fit_kwds.update(fkwds)
    history = defaultdict(list)
    idx_nonzero = np.arange(k_keep, dtype=int)
    keep = np.ones(k_keep, np.bool_)
    idx_excl = np.arange(k_keep, k_vars)
    mod_pen = model_class(endog, x0, **self.init_kwds)
    mod_pen.pen_weight = 0
    res_pen = mod_pen.fit(**fit_kwds)
    start_params = res_pen.params
    converged = False
    idx_old = []
    for it in range(maxiter):
        x1 = x[:, idx_excl]
        mom_cond = self.ranking_measure(res_pen, x1, keep=keep)
        assert len(mom_cond) == len(idx_excl)
        mcs = np.sort(mom_cond)[::-1]
        idx_thr = min((self.k_max_add, k_current + self.k_add, len(mcs)))
        threshold = mcs[idx_thr]
        idx = np.concatenate((idx_nonzero, idx_excl[mom_cond > threshold]))
        start_params2 = np.zeros(len(idx))
        start_params2[:len(start_params)] = start_params
        if self.use_weights:
            weights = np.ones(len(idx))
            weights[:k_keep] = 0
            self.penal.weights = weights
        mod_pen = model_class(endog, x[:, idx], penal=self.penal, pen_weight=self.pen_weight, **self.init_kwds)
        res_pen = mod_pen.fit(method=method, start_params=start_params2, warn_convergence=False, skip_hessian=True, **fit_kwds)
        keep = np.abs(res_pen.params) > self.threshold_trim
        if keep.sum() > self.k_max_included:
            thresh_params = np.sort(np.abs(res_pen.params))[-self.k_max_included]
            keep2 = np.abs(res_pen.params) > thresh_params
            keep = np.logical_and(keep, keep2)
        keep[:k_keep] = True
        idx_nonzero = idx[keep]
        if disp:
            print(keep)
            print(idx_nonzero)
        k_current = len(idx_nonzero)
        start_params = res_pen.params[keep]
        mask_excl = np.ones(k_vars, dtype=bool)
        mask_excl[idx_nonzero] = False
        idx_excl = np.nonzero(mask_excl)[0]
        history['idx_nonzero'].append(idx_nonzero)
        history['keep'].append(keep)
        history['params_keep'].append(start_params)
        history['idx_added'].append(idx)
        if len(idx_nonzero) == len(idx_old) and (idx_nonzero == idx_old).all():
            converged = True
            break
        idx_old = idx_nonzero
    assert np.all(idx_nonzero[:k_keep] == np.arange(k_keep))
    if self.use_weights:
        weights = np.ones(len(idx_nonzero))
        weights[:k_keep] = 0
        penal = self._get_penal(weights=weights)
    else:
        penal = self.penal
    mod_final = model_class(endog, x[:, idx_nonzero], penal=penal, pen_weight=self.pen_weight, **self.init_kwds)
    res_final = mod_final.fit(method=method, start_params=start_params, warn_convergence=False, **fit_kwds)
    xnames = ['var%4d' % ii for ii in idx_nonzero]
    res_final.model.exog_names[k_keep:] = xnames[k_keep:]
    res = ScreeningResults(self, results_pen=res_pen, results_final=res_final, idx_nonzero=idx_nonzero, idx_exog=idx_nonzero[k_keep:] - k_keep, idx_excl=idx_excl, history=history, converged=converged, iterations=it + 1)
    return res