import re, sys, os, tempfile, json
def phc_direct_base(var_names, eqns_as_strings):
    import cyphc
    mangled_vars = [remove_forbidden(v) for v in var_names]
    R = cyphc.PolyRing(mangled_vars)
    polys = [cyphc.PHCPoly(R, remove_forbidden(eqn)) for eqn in eqns_as_strings]
    system = cyphc.PHCSystem(R, polys)
    sols = system.solution_list()
    return [sol_to_dict(sol, var_names) for sol in sols]