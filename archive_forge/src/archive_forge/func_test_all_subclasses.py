from monty.inspect import all_subclasses, caller_name, find_top_pyfile
def test_all_subclasses(self):
    assert all_subclasses(LittleCatA) == [LittleCatB, LittleCatD]