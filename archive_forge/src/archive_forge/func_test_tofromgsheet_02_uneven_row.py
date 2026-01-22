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
def test_tofromgsheet_02_uneven_row():
    test_table_t1 = [x + ['3'] if i in [2] else x for i, x in enumerate(TEST_TABLE[:])]
    test_table_f1 = [x + [''] if len(x) < 3 else x for x in test_table_t1[:]]
    _test_to_fromg_sheet(test_table_t1, None, None, test_table_f1)