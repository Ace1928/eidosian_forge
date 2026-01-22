from operator import __getitem__
def testIReadMapping(self):
    inst = self._IReadMapping__sample()
    state = self._IReadMapping__stateDict()
    absent = self._IReadMapping__absentKeys()
    testIReadMapping(self, inst, state, absent)