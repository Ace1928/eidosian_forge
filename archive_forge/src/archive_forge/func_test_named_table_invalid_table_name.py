import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def test_named_table_invalid_table_name():
    test_table_1 = pa.Table.from_pydict({'x': [1, 2, 3]})

    def table_provider(names, _):
        if not names:
            raise Exception('No names provided')
        elif names[0] == 't1':
            return test_table_1
        else:
            raise Exception('Unrecognized table name')
    substrait_query = '\n    {\n        "version": { "major": 9999 },\n        "relations": [\n        {"rel": {\n            "read": {\n            "base_schema": {\n                "struct": {\n                "types": [\n                            {"i64": {}}\n                        ]\n                },\n                "names": [\n                        "x"\n                        ]\n            },\n            "namedTable": {\n                    "names": ["t3"]\n            }\n            }\n        }}\n        ]\n    }\n    '
    buf = pa._substrait._parse_json_plan(tobytes(substrait_query))
    exec_message = 'Invalid NamedTable Source'
    with pytest.raises(ArrowInvalid, match=exec_message):
        substrait.run_query(buf, table_provider=table_provider)