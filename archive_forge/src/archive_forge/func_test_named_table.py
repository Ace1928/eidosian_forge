import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
@pytest.mark.parametrize('use_threads', [True, False])
def test_named_table(use_threads):
    test_table_1 = pa.Table.from_pydict({'x': [1, 2, 3]})
    test_table_2 = pa.Table.from_pydict({'x': [4, 5, 6]})
    schema_1 = pa.schema([pa.field('x', pa.int64())])

    def table_provider(names, schema):
        if not names:
            raise Exception('No names provided')
        elif names[0] == 't1':
            assert schema == schema_1
            return test_table_1
        elif names[1] == 't2':
            return test_table_2
        else:
            raise Exception('Unrecognized table name')
    substrait_query = '\n    {\n        "version": { "major": 9999 },\n        "relations": [\n        {"rel": {\n            "read": {\n            "base_schema": {\n                "struct": {\n                "types": [\n                            {"i64": {}}\n                        ]\n                },\n                "names": [\n                        "x"\n                        ]\n            },\n            "namedTable": {\n                    "names": ["t1"]\n            }\n            }\n        }}\n        ]\n    }\n    '
    buf = pa._substrait._parse_json_plan(tobytes(substrait_query))
    reader = pa.substrait.run_query(buf, table_provider=table_provider, use_threads=use_threads)
    res_tb = reader.read_all()
    assert res_tb == test_table_1