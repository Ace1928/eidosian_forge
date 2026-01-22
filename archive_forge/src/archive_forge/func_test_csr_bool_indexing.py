import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix, hstack
import pytest
def test_csr_bool_indexing():
    data = csr_matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    list_indices1 = [False, True, False]
    array_indices1 = np.array(list_indices1)
    list_indices2 = [[False, True, False], [False, True, False], [False, True, False]]
    array_indices2 = np.array(list_indices2)
    list_indices3 = ([False, True, False], [False, True, False])
    array_indices3 = (np.array(list_indices3[0]), np.array(list_indices3[1]))
    slice_list1 = data[list_indices1].toarray()
    slice_array1 = data[array_indices1].toarray()
    slice_list2 = data[list_indices2]
    slice_array2 = data[array_indices2]
    slice_list3 = data[list_indices3]
    slice_array3 = data[array_indices3]
    assert (slice_list1 == slice_array1).all()
    assert (slice_list2 == slice_array2).all()
    assert (slice_list3 == slice_array3).all()