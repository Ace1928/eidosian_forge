from io import StringIO
def internQuery(queryString):
    if queryString not in __internedQueries:
        __internedQueries[queryString] = XPathQuery(queryString)
    return __internedQueries[queryString]