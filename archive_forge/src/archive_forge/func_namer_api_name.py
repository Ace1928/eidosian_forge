from collections import OrderedDict
def namer_api_name(inplace):
    if inplace:
        return 'rename_'
    else:
        return 'rename'