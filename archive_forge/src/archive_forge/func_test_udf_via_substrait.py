import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
@pytest.mark.parametrize('use_threads', [True, False])
def test_udf_via_substrait(unary_func_fixture, use_threads):
    test_table = pa.Table.from_pydict({'x': [1, 2, 3]})

    def table_provider(names, _):
        if not names:
            raise Exception('No names provided')
        elif names[0] == 't1':
            return test_table
        else:
            raise Exception('Unrecognized table name')
    substrait_query = b'\n    {\n  "extensionUris": [\n    {\n      "extensionUriAnchor": 1\n    },\n    {\n      "extensionUriAnchor": 2,\n      "uri": "urn:arrow:substrait_simple_extension_function"\n    }\n  ],\n  "extensions": [\n    {\n      "extensionFunction": {\n        "extensionUriReference": 2,\n        "functionAnchor": 1,\n        "name": "y=x+1"\n      }\n    }\n  ],\n  "relations": [\n    {\n      "root": {\n        "input": {\n          "project": {\n            "common": {\n              "emit": {\n                "outputMapping": [\n                  1,\n                  2,\n                ]\n              }\n            },\n            "input": {\n              "read": {\n                "baseSchema": {\n                  "names": [\n                    "t",\n                  ],\n                  "struct": {\n                    "types": [\n                      {\n                        "i64": {\n                          "nullability": "NULLABILITY_REQUIRED"\n                        }\n                      },\n                    ],\n                    "nullability": "NULLABILITY_REQUIRED"\n                  }\n                },\n                "namedTable": {\n                  "names": [\n                    "t1"\n                  ]\n                }\n              }\n            },\n            "expressions": [\n              {\n                "selection": {\n                  "directReference": {\n                    "structField": {}\n                  },\n                  "rootReference": {}\n                }\n              },\n              {\n                "scalarFunction": {\n                  "functionReference": 1,\n                  "outputType": {\n                    "i64": {\n                      "nullability": "NULLABILITY_NULLABLE"\n                    }\n                  },\n                  "arguments": [\n                    {\n                      "value": {\n                        "selection": {\n                          "directReference": {\n                            "structField": {}\n                          },\n                          "rootReference": {}\n                        }\n                      }\n                    }\n                  ]\n                }\n              }\n            ]\n          }\n        },\n        "names": [\n          "x",\n          "y",\n        ]\n      }\n    }\n  ]\n}\n    '
    buf = pa._substrait._parse_json_plan(substrait_query)
    reader = pa.substrait.run_query(buf, table_provider=table_provider, use_threads=use_threads)
    res_tb = reader.read_all()
    function, name = unary_func_fixture
    expected_tb = test_table.add_column(1, 'y', function(mock_udf_context(10), test_table['x']))
    assert res_tb == expected_tb