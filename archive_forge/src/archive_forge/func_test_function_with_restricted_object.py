import dill
def test_function_with_restricted_object():
    deserialized = dill.loads(dill.dumps(restricted_func, recurse=True))