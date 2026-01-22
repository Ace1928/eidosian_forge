from monty.itertools import chunks, iterator_from_slice
def test_iterator_from_slice():
    assert list(iterator_from_slice(slice(0, 6, 2))) == [0, 2, 4]