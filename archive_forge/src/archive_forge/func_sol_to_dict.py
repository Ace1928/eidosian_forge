import re, sys, os, tempfile, json
def sol_to_dict(sol, vars):
    ans = {v: clean_complex(p) for v, p in zip(vars, sol.point)}
    for attr in ['err', 'rco', 'res', 'mult']:
        ans[attr] = getattr(sol, attr)
    return ans