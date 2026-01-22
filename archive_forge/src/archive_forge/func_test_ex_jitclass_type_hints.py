import unittest
from numba.tests.support import TestCase
def test_ex_jitclass_type_hints(self):
    from typing import List
    from numba.experimental import jitclass
    from numba.typed import List as NumbaList

    @jitclass
    class Counter:
        value: int

        def __init__(self):
            self.value = 0

        def get(self) -> int:
            ret = self.value
            self.value += 1
            return ret

    @jitclass
    class ListLoopIterator:
        counter: Counter
        items: List[float]

        def __init__(self, items: List[float]):
            self.items = items
            self.counter = Counter()

        def get(self) -> float:
            idx = self.counter.get() % len(self.items)
            return self.items[idx]
    items = NumbaList([3.14, 2.718, 0.123, -4.0])
    loop_itr = ListLoopIterator(items)
    for idx in range(10):
        self.assertEqual(loop_itr.counter.value, idx)
        self.assertAlmostEqual(loop_itr.get(), items[idx % len(items)])
        self.assertEqual(loop_itr.counter.value, idx + 1)