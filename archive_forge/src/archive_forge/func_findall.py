import re
def findall(elem, path, namespaces=None, with_prefixes=True):
    return list(iterfind(elem, path, namespaces))