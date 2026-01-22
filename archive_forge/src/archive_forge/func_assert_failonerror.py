import pytest
from petl.test.helpers import ieq, eq_
import petl.config as config
def assert_failonerror(input_fn, expected_output):
    """In the input rows, the first row should process through the
    transformation cleanly.  The second row should generate an
    exception.  There are no requirements for any other rows."""
    eq_(config.failonerror, False)
    table2 = input_fn()
    ieq(expected_output, table2)
    ieq(expected_output, table2)
    table3 = input_fn(failonerror=False)
    ieq(expected_output, table3)
    ieq(expected_output, table3)
    table3 = input_fn(failonerror=None)
    ieq(expected_output, table3)
    ieq(expected_output, table3)
    with pytest.raises(Exception):
        table4 = input_fn(failonerror=True)
        table4.nrows()
    expect5 = (expected_output[0], expected_output[1])
    table5 = input_fn(failonerror='inline')
    ieq(expect5, table5.head(1))
    ieq(expect5, table5.head(1))
    excp = table5[2][0]
    assert isinstance(excp, Exception)
    saved_config_failonerror = config.failonerror
    config.failonerror = True
    with pytest.raises(Exception):
        table6 = input_fn()
        table6.nrows()
    expect7 = (expected_output[0], expected_output[1])
    config.failonerror = 'inline'
    table7 = input_fn()
    ieq(expect7, table7.head(1))
    ieq(expect7, table7.head(1))
    excp = table7[2][0]
    assert isinstance(excp, Exception)
    config.failonerror = 'invalid'
    with pytest.raises(Exception):
        table8 = input_fn()
        table8.nrows()
    config.failonerror = None
    table9 = input_fn()
    ieq(expected_output, table9)
    ieq(expected_output, table9)
    config.failonerror = True
    table10 = input_fn(failonerror=False)
    ieq(expected_output, table10)
    ieq(expected_output, table10)
    config.failonerror = True
    with pytest.raises(Exception):
        table11 = input_fn(failonerror=None)
        table11.nrows()
    config.failonerror = saved_config_failonerror