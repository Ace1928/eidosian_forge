from __future__ import absolute_import, print_function, division
import datetime
import os
import json
import time
import pytest
from petl.compat import text_type
from petl.io.gsheet import fromgsheet, togsheet, appendgsheet
from petl.test.helpers import ieq, get_env_vars_named
@found_gcp_credentials
def test_tofromgsheet_04_cell_range():
    test_table_f2 = [[x[1]] for x in TEST_TABLE[0:4]]
    _test_to_fromg_sheet(TEST_TABLE[:], None, 'B1:B4', test_table_f2)