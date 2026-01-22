import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
@pytest.mark.parametrize('use_threads', [True, False])
def test_run_serialized_query(tmpdir, use_threads):
    substrait_query = '\n    {\n        "version": { "major": 9999 },\n        "relations": [\n        {"rel": {\n            "read": {\n            "base_schema": {\n                "struct": {\n                "types": [\n                            {"i64": {}}\n                        ]\n                },\n                "names": [\n                        "foo"\n                        ]\n            },\n            "local_files": {\n                "items": [\n                {\n                    "uri_file": "FILENAME_PLACEHOLDER",\n                    "arrow": {}\n                }\n                ]\n            }\n            }\n        }}\n        ]\n    }\n    '
    file_name = 'read_data.arrow'
    table = pa.table([[1, 2, 3, 4, 5]], names=['foo'])
    path = _write_dummy_data_to_disk(tmpdir, file_name, table)
    query = tobytes(substrait_query.replace('FILENAME_PLACEHOLDER', pathlib.Path(path).as_uri()))
    buf = pa._substrait._parse_json_plan(query)
    reader = substrait.run_query(buf, use_threads=use_threads)
    res_tb = reader.read_all()
    assert table.select(['foo']) == res_tb.select(['foo'])