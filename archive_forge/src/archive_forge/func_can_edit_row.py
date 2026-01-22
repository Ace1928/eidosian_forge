import numpy as np
import pandas
import pytest
from modin_spreadsheet import SpreadsheetWidget
import modin.experimental.spreadsheet as mss
import modin.pandas as pd
def can_edit_row(row):
    return row['D'] > 2