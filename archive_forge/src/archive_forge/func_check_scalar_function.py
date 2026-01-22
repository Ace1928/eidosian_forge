import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def check_scalar_function(func_fixture, inputs, *, run_in_dataset=True, batch_length=None):
    function, name = func_fixture
    if batch_length is None:
        all_scalar = True
        for arg in inputs:
            if isinstance(arg, pa.Array):
                all_scalar = False
                batch_length = len(arg)
        if all_scalar:
            batch_length = 1
    func = pc.get_function(name)
    assert func.name == name
    result = pc.call_function(name, inputs, length=batch_length)
    expected_output = function(mock_udf_context(batch_length), *inputs)
    assert result == expected_output
    if run_in_dataset:
        field_names = [f'field{index}' for index, in_arr in inputs]
        table = pa.Table.from_arrays(inputs, field_names)
        dataset = ds.dataset(table)
        func_args = [ds.field(field_name) for field_name in field_names]
        result_table = dataset.to_table(columns={'result': ds.field('')._call(name, func_args)})
        assert result_table.column(0).chunks[0] == expected_output