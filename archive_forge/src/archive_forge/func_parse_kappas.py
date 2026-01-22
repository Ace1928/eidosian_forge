import re
def parse_kappas(lines, parameters):
    """Parse out the kappa parameters."""
    kappa_found = False
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if 'Parameters (kappa)' in line:
            kappa_found = True
        elif kappa_found and line_floats:
            branch_res = re.match('\\s(\\d+\\.\\.\\d+)', line)
            if branch_res is None:
                if len(line_floats) == 1:
                    parameters['kappa'] = line_floats[0]
                else:
                    parameters['kappa'] = line_floats
                kappa_found = False
            else:
                if parameters.get('branches') is None:
                    parameters['branches'] = {}
                branch = branch_res.group(1)
                if line_floats:
                    parameters['branches'][branch] = {'t': line_floats[0], 'kappa': line_floats[1], 'TS': line_floats[2], 'TV': line_floats[3]}
        elif 'kappa under' in line and line_floats:
            if len(line_floats) == 1:
                parameters['kappa'] = line_floats[0]
            else:
                parameters['kappa'] = line_floats
    return parameters