import re
def parse_rates(lines, parameters):
    """Parse the rate parameters."""
    Q_mat_found = False
    trans_probs_found = False
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if 'Rate parameters:' in line and line_floats:
            parameters['rate parameters'] = line_floats
        elif 'rate: ' in line and line_floats:
            parameters['rates'] = line_floats
        elif 'matrix Q' in line:
            parameters['Q matrix'] = {'matrix': []}
            if line_floats:
                parameters['Q matrix']['average Ts/Tv'] = line_floats[0]
            Q_mat_found = True
        elif Q_mat_found and line_floats:
            parameters['Q matrix']['matrix'].append(line_floats)
            if len(parameters['Q matrix']['matrix']) == 4:
                Q_mat_found = False
        elif 'alpha' in line and line_floats:
            parameters['alpha'] = line_floats[0]
        elif 'rho' in line and line_floats:
            parameters['rho'] = line_floats[0]
        elif 'transition probabilities' in line:
            parameters['transition probs.'] = []
            trans_probs_found = True
        elif trans_probs_found and line_floats:
            parameters['transition probs.'].append(line_floats)
            if len(parameters['transition probs.']) == len(parameters['rates']):
                trans_probs_found = False
    return parameters