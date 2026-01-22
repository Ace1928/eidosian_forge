from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_header_multi_index(all_parsers):
    parser = all_parsers
    data = 'C0,,C_l0_g0,C_l0_g1,C_l0_g2\n\nC1,,C_l1_g0,C_l1_g1,C_l1_g2\nC2,,C_l2_g0,C_l2_g1,C_l2_g2\nC3,,C_l3_g0,C_l3_g1,C_l3_g2\nR0,R1,,,\nR_l0_g0,R_l1_g0,R0C0,R0C1,R0C2\nR_l0_g1,R_l1_g1,R1C0,R1C1,R1C2\nR_l0_g2,R_l1_g2,R2C0,R2C1,R2C2\nR_l0_g3,R_l1_g3,R3C0,R3C1,R3C2\nR_l0_g4,R_l1_g4,R4C0,R4C1,R4C2\n'
    result = parser.read_csv(StringIO(data), header=[0, 1, 2, 3], index_col=[0, 1])
    data_gen_f = lambda r, c: f'R{r}C{c}'
    data = [[data_gen_f(r, c) for c in range(3)] for r in range(5)]
    index = MultiIndex.from_arrays([[f'R_l0_g{i}' for i in range(5)], [f'R_l1_g{i}' for i in range(5)]], names=['R0', 'R1'])
    columns = MultiIndex.from_arrays([[f'C_l0_g{i}' for i in range(3)], [f'C_l1_g{i}' for i in range(3)], [f'C_l2_g{i}' for i in range(3)], [f'C_l3_g{i}' for i in range(3)]], names=['C0', 'C1', 'C2', 'C3'])
    expected = DataFrame(data, columns=columns, index=index)
    tm.assert_frame_equal(result, expected)