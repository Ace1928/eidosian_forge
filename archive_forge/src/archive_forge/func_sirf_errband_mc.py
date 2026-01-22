import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
def sirf_errband_mc(self, orth=False, repl=1000, steps=10, signif=0.05, seed=None, burn=100, cum=False):
    """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lütkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors
        """
    neqs = self.neqs
    mean = self.mean()
    k_ar = self.k_ar
    coefs = self.coefs
    sigma_u = self.sigma_u
    intercept = self.intercept
    df_model = self.df_model
    nobs = self.nobs
    ma_coll = np.zeros((repl, steps + 1, neqs, neqs))
    A = self.A
    B = self.B
    A_mask = self.A_mask
    B_mask = self.B_mask
    A_pass = self.model.A_original
    B_pass = self.model.B_original
    s_type = self.model.svar_type
    g_list = []

    def agg(impulses):
        if cum:
            return impulses.cumsum(axis=0)
        return impulses
    opt_A = A[A_mask]
    opt_B = B[B_mask]
    for i in range(repl):
        sim = util.varsim(coefs, intercept, sigma_u, seed=seed, steps=nobs + burn)
        sim = sim[burn:]
        smod = SVAR(sim, svar_type=s_type, A=A_pass, B=B_pass)
        if i == 10:
            mean_AB = np.mean(g_list, axis=0)
            split = len(A[A_mask])
            opt_A = mean_AB[:split]
            opt_B = mean_AB[split:]
        sres = smod.fit(maxlags=k_ar, A_guess=opt_A, B_guess=opt_B)
        if i < 10:
            g_list.append(np.append(sres.A[A_mask].tolist(), sres.B[B_mask].tolist()))
        ma_coll[i] = agg(sres.svar_ma_rep(maxn=steps))
    ma_sort = np.sort(ma_coll, axis=0)
    index = (int(round(signif / 2 * repl) - 1), int(round((1 - signif / 2) * repl) - 1))
    lower = ma_sort[index[0], :, :, :]
    upper = ma_sort[index[1], :, :, :]
    return (lower, upper)