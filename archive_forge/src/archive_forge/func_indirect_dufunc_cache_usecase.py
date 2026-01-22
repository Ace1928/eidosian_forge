import numba as nb
def indirect_dufunc_cache_usecase(**kwargs):

    @nb.njit(cache=True)
    def indirect_ufunc_core(inp):
        return inp * 3

    @nb.vectorize(**kwargs)
    def ufunc(inp):
        return indirect_ufunc_core(inp)
    return ufunc