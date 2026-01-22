from __future__ import (absolute_import, division, print_function)
def get_num_records(response):
    """ num_records is not always present
        if absent, count the records or assume 1
    """
    if 'num_records' in response:
        return response['num_records']
    return len(response['records']) if 'records' in response else 1