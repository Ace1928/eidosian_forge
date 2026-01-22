import re, sys, os, tempfile, json
def serialize_sol_dict(sol):
    sol = sol.copy()
    for key, val in sol.items():
        if isinstance(val, complex):
            sol[key] = (val.real, val.imag)
    return sol