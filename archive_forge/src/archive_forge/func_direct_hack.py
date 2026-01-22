import re, sys, os, tempfile, json
def direct_hack(backend, ideal, **kwargs):
    """
    To avoid memory leaks and random PARI crashes, runs CyPHC
    in a separate subprocess.
    """
    vars = ideal.ring().variable_names()
    polys = [repr(eqn) for eqn in ideal.gens()]
    data = {'backend': backend, 'vars': vars, 'polys': polys, 'kwargs': kwargs}
    problem_data = json.dumps(data).encode('base64').replace('\n', '')
    ans_data = os.popen('sage -python ' + __file__ + ' ' + problem_data).read()
    ans_data = re.sub('PHCv.*? released .*? works!\n', '', ans_data)
    if len(ans_data):
        ans = json.loads(ans_data)
        for sol in ans:
            for key, val in sol.items():
                if isinstance(val, list):
                    sol[key] = complex(*val)
    else:
        ans = []
    return ans