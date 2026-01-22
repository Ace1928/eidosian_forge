import numpy as np
import numpy.lib.recfunctions
from statsmodels.compat.python import lmap
from statsmodels.regression.linear_model import OLS
def form2design(ss, data):
    """convert string formula to data dictionary

    ss : str
     * I : add constant
     * varname : for simple varnames data is used as is
     * F:varname : create dummy variables for factor varname
     * P:varname1*varname2 : create product dummy variables for
       varnames
     * G:varname1*varname2 : create product between factor and
       continuous variable
    data : dict or structured array
       data set, access of variables by name as in dictionaries

    Returns
    -------
    vars : dictionary
        dictionary of variables with converted dummy variables
    names : list
        list of names, product (P:) and grouped continuous
        variables (G:) have name by joining individual names
        sorted according to input

    Examples
    --------
    >>> xx, n = form2design('I a F:b P:c*d G:c*f', testdata)
    >>> xx.keys()
    ['a', 'b', 'const', 'cf', 'cd']
    >>> n
    ['const', 'a', 'b', 'cd', 'cf']

    Notes
    -----

    with sorted dict, separate name list would not be necessary
    """
    vars = {}
    names = []
    for item in ss.split():
        if item == 'I':
            vars['const'] = np.ones(data.shape[0])
            names.append('const')
        elif ':' not in item:
            vars[item] = data[item]
            names.append(item)
        elif item[:2] == 'F:':
            v = item.split(':')[1]
            vars[v] = data2dummy(data[v])
            names.append(v)
        elif item[:2] == 'P:':
            v = item.split(':')[1].split('*')
            vars[''.join(v)] = data2proddummy(np.c_[data[v[0]], data[v[1]]])
            names.append(''.join(v))
        elif item[:2] == 'G:':
            v = item.split(':')[1].split('*')
            vars[''.join(v)] = data2groupcont(data[v[0]], data[v[1]])
            names.append(''.join(v))
        else:
            raise ValueError('unknown expression in formula')
    return (vars, names)