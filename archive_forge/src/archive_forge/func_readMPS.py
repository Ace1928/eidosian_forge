import re
from . import constants as const
def readMPS(path, sense, dropConsNames=False):
    """
    adapted from Julian Märte (https://github.com/pchtsp/pysmps)
    returns a dictionary with the contents of the model.
    This dictionary can be used to generate an LpProblem

    :param path: path of mps file
    :param sense: 1 for minimize, -1 for maximize
    :param dropConsNames: if True, do not store the names of constraints
    :return: a dictionary with all the problem data
    """
    mode = ''
    parameters = dict(name='', sense=sense, status=0, sol_status=0)
    variable_info = {}
    constraints = {}
    objective = dict(name='', coefficients=[])
    sos1 = []
    sos2 = []
    rhs_names = []
    bnd_names = []
    integral_marker = False
    with open(path) as reader:
        for line in reader:
            line = re.split(' |\t', line)
            line = [x.strip() for x in line]
            line = list(filter(None, line))
            if line[0] == 'ENDATA':
                break
            if line[0] == '*':
                continue
            if line[0] == 'NAME':
                if len(line) > 1:
                    parameters['name'] = line[1]
                else:
                    parameters['name'] = ''
                continue
            if line[0] in [CORE_FILE_ROW_MODE, CORE_FILE_COL_MODE]:
                mode = line[0]
            elif line[0] == CORE_FILE_RHS_MODE and len(line) <= 2:
                if len(line) > 1:
                    rhs_names.append(line[1])
                    mode = CORE_FILE_RHS_MODE_NAME_GIVEN
                else:
                    mode = CORE_FILE_RHS_MODE_NO_NAME
            elif line[0] == CORE_FILE_BOUNDS_MODE and len(line) <= 2:
                if len(line) > 1:
                    bnd_names.append(line[1])
                    mode = CORE_FILE_BOUNDS_MODE_NAME_GIVEN
                else:
                    mode = CORE_FILE_BOUNDS_MODE_NO_NAME
            elif mode == CORE_FILE_ROW_MODE:
                row_type = line[0]
                row_name = line[1]
                if row_type == ROW_MODE_OBJ:
                    objective['name'] = row_name
                else:
                    constraints[row_name] = dict(sense=ROW_EQUIV[row_type], name=row_name, coefficients=[], **ROW_DEFAULT)
            elif mode == CORE_FILE_COL_MODE:
                var_name = line[0]
                if len(line) > 1 and line[1] == "'MARKER'":
                    if line[2] == "'INTORG'":
                        integral_marker = True
                    elif line[2] == "'INTEND'":
                        integral_marker = False
                    continue
                if var_name not in variable_info:
                    variable_info[var_name] = dict(cat=COL_EQUIV[integral_marker], name=var_name, **COL_DEFAULT)
                j = 1
                while j < len(line) - 1:
                    if line[j] == objective['name']:
                        objective['coefficients'].append(dict(name=var_name, value=float(line[j + 1])))
                    else:
                        constraints[line[j]]['coefficients'].append(dict(name=var_name, value=float(line[j + 1])))
                    j = j + 2
            elif mode == CORE_FILE_RHS_MODE_NAME_GIVEN:
                if line[0] != rhs_names[-1]:
                    raise Exception('Other RHS name was given even though name was set after RHS tag.')
                readMPSSetRhs(line, constraints)
            elif mode == CORE_FILE_RHS_MODE_NO_NAME:
                readMPSSetRhs(line, constraints)
                if line[0] not in rhs_names:
                    rhs_names.append(line[0])
            elif mode == CORE_FILE_BOUNDS_MODE_NAME_GIVEN:
                if line[1] != bnd_names[-1]:
                    raise Exception('Other BOUNDS name was given even though name was set after BOUNDS tag.')
                readMPSSetBounds(line, variable_info)
            elif mode == CORE_FILE_BOUNDS_MODE_NO_NAME:
                readMPSSetBounds(line, variable_info)
                if line[1] not in bnd_names:
                    bnd_names.append(line[1])
    constraints = list(constraints.values())
    if dropConsNames:
        for c in constraints:
            c['name'] = None
        objective['name'] = None
    variable_info = list(variable_info.values())
    return dict(parameters=parameters, objective=objective, variables=variable_info, constraints=constraints, sos1=sos1, sos2=sos2)