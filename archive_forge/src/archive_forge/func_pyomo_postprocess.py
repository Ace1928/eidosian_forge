from . import pmedian
def pyomo_postprocess(**kwds):
    print('POSTPROCESSING %s' % sorted(list(kwds.keys())))