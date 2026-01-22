import abc
import uuid
from typing import List, Tuple
import numpy as np
import pyarrow as pa
from modin.error_message import ErrorMessage
@classmethod
def cast_to_compatible_types(cls, table, cast_dict):
    """
        Cast PyArrow table to be fully compatible with HDK.

        Parameters
        ----------
        table : pyarrow.Table
            Source table.
        cast_dict : bool
            Cast dictionary columns to string.

        Returns
        -------
        pyarrow.Table
            Table with fully compatible types with HDK.
        """
    schema = table.schema
    new_schema = schema
    need_cast = False
    uint_to_int_cast = False
    for i, field in enumerate(schema):
        if pa.types.is_dictionary(field.type):
            value_type = field.type.value_type
            if pa.types.is_null(value_type):
                mask = np.full(table.num_rows, True, dtype=bool)
                new_col_data = np.empty(table.num_rows, dtype=str)
                new_col = pa.array(new_col_data, pa.string(), mask)
                new_field = pa.field(field.name, pa.string(), field.nullable, field.metadata)
                table = table.set_column(i, new_field, new_col)
            elif pa.types.is_string(value_type):
                if cast_dict:
                    need_cast = True
                    new_field = pa.field(field.name, pa.string(), field.nullable, field.metadata)
                else:
                    new_field = field
            else:
                new_field, int_cast = cls._convert_field(field, value_type)
                need_cast = True
                uint_to_int_cast = uint_to_int_cast or int_cast
                if new_field == field:
                    new_field = pa.field(field.name, value_type, field.nullable, field.metadata)
            new_schema = new_schema.set(i, new_field)
        else:
            new_field, int_cast = cls._convert_field(field, field.type)
            need_cast = need_cast or new_field is not field
            uint_to_int_cast = uint_to_int_cast or int_cast
            new_schema = new_schema.set(i, new_field)
    if uint_to_int_cast:
        ErrorMessage.single_warning('HDK does not support unsigned integer types, such types will be rounded up to the signed equivalent.')
    if need_cast:
        try:
            table = table.cast(new_schema)
        except pa.lib.ArrowInvalid as err:
            raise (OverflowError if uint_to_int_cast else RuntimeError)("An error occurred when trying to convert unsupported by HDK 'dtypes' " + f'to the supported ones, the schema to cast was: \n{new_schema}.') from err
    return table