import itertools
import collections
def makeDict(headers, array, default=None):
    """
    makes a list into a dictionary with the headings given in headings
    headers is a list of header lists
    array is a list with the data
    """
    result, defdict = __makeDict(headers, array, default)
    return result