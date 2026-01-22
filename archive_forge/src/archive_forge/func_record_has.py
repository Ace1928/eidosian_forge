import copy
def record_has(inrec, fieldvals):
    """Accept a record, and a dictionary of field values.

    The format is {'field_name': set([val1, val2])}.
    If any field in the record has  a matching value, the function returns
    True. Otherwise, returns False.
    """
    retval = False
    for field in fieldvals:
        if isinstance(inrec[field], str):
            set1 = {inrec[field]}
        else:
            set1 = set(inrec[field])
        if set1 & fieldvals[field]:
            retval = True
            break
    return retval