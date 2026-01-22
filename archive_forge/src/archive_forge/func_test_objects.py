import dill as pickle
from dill import load_types, objects, extend
def test_objects(verbose=True):
    for member in objects.keys():
        pickles(member, exact=False, verbose=verbose)