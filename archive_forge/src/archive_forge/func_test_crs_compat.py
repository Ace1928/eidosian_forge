import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def test_crs_compat():
    from patsy.test_state import check_stateful
    from patsy.test_splines_crs_data import R_crs_test_x, R_crs_test_data, R_crs_num_tests
    lines = R_crs_test_data.split('\n')
    tests_ran = 0
    start_idx = lines.index('--BEGIN TEST CASE--')
    while True:
        if not lines[start_idx] == '--BEGIN TEST CASE--':
            break
        start_idx += 1
        stop_idx = lines.index('--END TEST CASE--', start_idx)
        block = lines[start_idx:stop_idx]
        test_data = {}
        for line in block:
            key, value = line.split('=', 1)
            test_data[key] = value
        adjust_df = 0
        if test_data['spline_type'] == 'cr' or test_data['spline_type'] == 'cs':
            spline_type = CR
        elif test_data['spline_type'] == 'cc':
            spline_type = CC
            adjust_df += 1
        else:
            raise ValueError('Unrecognized spline type %r' % (test_data['spline_type'],))
        kwargs = {}
        if test_data['absorb_cons'] == 'TRUE':
            kwargs['constraints'] = 'center'
            adjust_df += 1
        if test_data['knots'] != 'None':
            all_knots = np.asarray(eval(test_data['knots']))
            all_knots.sort()
            kwargs['knots'] = all_knots[1:-1]
            kwargs['lower_bound'] = all_knots[0]
            kwargs['upper_bound'] = all_knots[-1]
        else:
            kwargs['df'] = eval(test_data['nb_knots']) - adjust_df
        output = np.asarray(eval(test_data['output']))
        check_stateful(spline_type, False, R_crs_test_x, output, **kwargs)
        tests_ran += 1
        start_idx = stop_idx + 1
    assert tests_ran == R_crs_num_tests