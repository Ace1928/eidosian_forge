from . import pmedian
def pyomo_print_model(**kwds):
    print('PRINTING MODEL %s' % sorted(list(kwds.keys())))