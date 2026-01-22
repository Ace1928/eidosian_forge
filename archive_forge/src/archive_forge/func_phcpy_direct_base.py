import re, sys, os, tempfile, json
def phcpy_direct_base(var_names, eqns_as_strings, tasks=0, precision='d', tolerance=1e-06):
    """
    Use the official PHCPack Python interface to find solutions to the
    given equations. To duplicate the behavior of CyPHC, we filter out
    any solutions with a coordinate too close to zero or infinity, as
    determined by the tolerance parameter.  It's not clear to me why
    this is needed since I believe phcpy is calling the Laurent
    polynomial version of it's solver, which presumably assumes this,
    but...
    """
    import phcpy
    polys = [remove_forbidden(eqn) + ';' for eqn in eqns_as_strings]
    sols = phcpy.solver.solve(polys, verbose=False, tasks=tasks, precision=precision, checkin=True)
    ans = []
    for sol in sols:
        if sol.find('NaN******') > -1:
            continue
        sol = phcpy.solutions.strsol2dict(sol)
        good_sol = True
        sol['mult'] = sol['m']
        sol.pop('m')
        sol['t_hom_val'] = sol['t']
        sol.pop('t')
        for v in var_names:
            w = remove_forbidden(v)
            if v != w:
                sol[v] = sol[w]
                sol.pop(w)
            sol[v] = clean_complex(sol[v])
            size = abs(sol[v])
            if size < tolerance or size > 1.0 / tolerance:
                good_sol = False
        if good_sol:
            ans.append(sol)
    return ans