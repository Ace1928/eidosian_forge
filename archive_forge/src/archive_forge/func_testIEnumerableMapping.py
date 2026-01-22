from operator import __getitem__
def testIEnumerableMapping(self, inst, state):
    test_keys(self, inst, state)
    test_items(self, inst, state)
    test_values(self, inst, state)
    test___len__(self, inst, state)