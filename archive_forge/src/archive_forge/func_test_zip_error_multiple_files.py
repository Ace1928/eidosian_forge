import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('compression', ['zip', 'infer'])
def test_zip_error_multiple_files(parser_and_data, compression):
    parser, data, expected = parser_and_data
    with tm.ensure_clean('combined_zip.zip') as path:
        inner_file_names = ['test_file', 'second_file']
        with zipfile.ZipFile(path, mode='w') as tmp:
            for file_name in inner_file_names:
                tmp.writestr(file_name, data)
        with pytest.raises(ValueError, match='Multiple files'):
            parser.read_csv(path, compression=compression)