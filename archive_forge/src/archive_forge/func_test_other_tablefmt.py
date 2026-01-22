from io import (
import pytest
import pandas as pd
import pandas._testing as tm
def test_other_tablefmt():
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, tablefmt='jira')
    result = buf.getvalue()
    assert result == '||    ||   0 ||\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |'