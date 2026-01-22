def has_column_major_storage(arr):
    return arr.flags['FORTRAN']