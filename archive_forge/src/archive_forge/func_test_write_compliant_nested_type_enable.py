import pytest
import pyarrow as pa
@pytest.mark.pandas
@parametrize_test_data
def test_write_compliant_nested_type_enable(tempdir, test_data):
    df = pd.DataFrame(data=test_data)
    _roundtrip_pandas_dataframe(df, write_kwargs={})
    table = pa.Table.from_pandas(df, preserve_index=False)
    path = str(tempdir / 'data.parquet')
    with pq.ParquetWriter(path, table.schema, version='2.6') as writer:
        writer.write_table(table)
    new_table = _read_table(path)
    assert isinstance(new_table.schema.types[0], pa.ListType)
    assert new_table.schema.types[0].value_field.name == 'element'
    _check_roundtrip(new_table)