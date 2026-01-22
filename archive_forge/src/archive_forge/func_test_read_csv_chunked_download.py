from io import BytesIO
import logging
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv
@pytest.mark.single_cpu
def test_read_csv_chunked_download(self, s3_public_bucket, caplog, s3so):
    df = DataFrame(np.zeros((100000, 4)), columns=list('abcd'))
    with BytesIO(df.to_csv().encode('utf-8')) as buf:
        s3_public_bucket.put_object(Key='large-file.csv', Body=buf)
        uri = f'{s3_public_bucket.name}/large-file.csv'
        match_re = re.compile(f'^Fetch: {uri}, 0-(?P<stop>\\d+)$')
        with caplog.at_level(logging.DEBUG, logger='s3fs'):
            read_csv(f's3://{uri}', nrows=5, storage_options=s3so)
            for log in caplog.messages:
                if (match := re.match(match_re, log)):
                    assert int(match.group('stop')) < 8000000