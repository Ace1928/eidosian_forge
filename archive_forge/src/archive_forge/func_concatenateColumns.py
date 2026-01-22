import numpy as np
from ...metaarray import MetaArray
def concatenateColumns(data):
    """Returns a single record array with columns taken from the elements in data. 
    data should be a list of elements, which can be either record arrays or tuples (name, type, data)
    """
    dtype = []
    names = set()
    maxLen = 0
    for element in data:
        if isinstance(element, np.ndarray):
            for i in range(len(element.dtype)):
                name = element.dtype.names[i]
                dtype.append((name, element.dtype[i]))
            maxLen = max(maxLen, len(element))
        else:
            name, type, d = element
            if type is None:
                type = suggestDType(d)
            dtype.append((name, type))
            if isinstance(d, list) or isinstance(d, np.ndarray):
                maxLen = max(maxLen, len(d))
        if name in names:
            raise Exception('Name "%s" repeated' % name)
        names.add(name)
    out = np.empty(maxLen, dtype)
    for element in data:
        if isinstance(element, np.ndarray):
            for i in range(len(element.dtype)):
                name = element.dtype.names[i]
                try:
                    out[name] = element[name]
                except:
                    print('Column:', name)
                    print('Input shape:', element.shape, element.dtype)
                    print('Output shape:', out.shape, out.dtype)
                    raise
        else:
            name, type, d = element
            out[name] = d
    return out