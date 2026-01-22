import io
import csv
import logging
from petl.util.base import Table, data
def fromcsv_impl(source, **kwargs):
    return CSVView(source, **kwargs)