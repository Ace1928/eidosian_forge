import os
def series_to_line(row, sep):
    return sep.join(map(str, row.tolist()))