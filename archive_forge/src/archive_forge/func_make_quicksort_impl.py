import collections
import numpy as np
from numba.core import types
def make_quicksort_impl(wrap, lt=None, is_argsort=False, is_list=False, is_np_array=False):
    intp = types.intp
    zero = intp(0)
    if is_argsort:
        if is_list:

            @wrap
            def make_res(A):
                return [x for x in range(len(A))]
        else:

            @wrap
            def make_res(A):
                return np.arange(A.size)

        @wrap
        def GET(A, idx_or_val):
            return A[idx_or_val]
    else:

        @wrap
        def make_res(A):
            return A

        @wrap
        def GET(A, idx_or_val):
            return idx_or_val

    def default_lt(a, b):
        """
        Trivial comparison function between two keys.
        """
        return a < b
    LT = wrap(lt if lt is not None else default_lt)

    @wrap
    def insertion_sort(A, R, low, high):
        """
        Insertion sort A[low:high + 1]. Note the inclusive bounds.
        """
        assert low >= 0
        if high <= low:
            return
        for i in range(low + 1, high + 1):
            k = R[i]
            v = GET(A, k)
            j = i
            while j > low and LT(v, GET(A, R[j - 1])):
                R[j] = R[j - 1]
                j -= 1
            R[j] = k

    @wrap
    def partition(A, R, low, high):
        """
        Partition A[low:high + 1] around a chosen pivot.  The pivot's index
        is returned.
        """
        assert low >= 0
        assert high > low
        mid = low + high >> 1
        if LT(GET(A, R[mid]), GET(A, R[low])):
            R[low], R[mid] = (R[mid], R[low])
        if LT(GET(A, R[high]), GET(A, R[mid])):
            R[high], R[mid] = (R[mid], R[high])
        if LT(GET(A, R[mid]), GET(A, R[low])):
            R[low], R[mid] = (R[mid], R[low])
        pivot = GET(A, R[mid])
        R[high], R[mid] = (R[mid], R[high])
        i = low
        j = high - 1
        while True:
            while i < high and LT(GET(A, R[i]), pivot):
                i += 1
            while j >= low and LT(pivot, GET(A, R[j])):
                j -= 1
            if i >= j:
                break
            R[i], R[j] = (R[j], R[i])
            i += 1
            j -= 1
        R[i], R[high] = (R[high], R[i])
        return i

    @wrap
    def partition3(A, low, high):
        """
        Three-way partition [low, high) around a chosen pivot.
        A tuple (lt, gt) is returned such that:
            - all elements in [low, lt) are < pivot
            - all elements in [lt, gt] are == pivot
            - all elements in (gt, high] are > pivot
        """
        mid = low + high >> 1
        if LT(A[mid], A[low]):
            A[low], A[mid] = (A[mid], A[low])
        if LT(A[high], A[mid]):
            A[high], A[mid] = (A[mid], A[high])
        if LT(A[mid], A[low]):
            A[low], A[mid] = (A[mid], A[low])
        pivot = A[mid]
        A[low], A[mid] = (A[mid], A[low])
        lt = low
        gt = high
        i = low + 1
        while i <= gt:
            if LT(A[i], pivot):
                A[lt], A[i] = (A[i], A[lt])
                lt += 1
                i += 1
            elif LT(pivot, A[i]):
                A[gt], A[i] = (A[i], A[gt])
                gt -= 1
            else:
                i += 1
        return (lt, gt)

    @wrap
    def run_quicksort1(A):
        R = make_res(A)
        if len(A) < 2:
            return R
        stack = [Partition(zero, zero)] * MAX_STACK
        stack[0] = Partition(zero, len(A) - 1)
        n = 1
        while n > 0:
            n -= 1
            low, high = stack[n]
            while high - low >= SMALL_QUICKSORT:
                assert n < MAX_STACK
                i = partition(A, R, low, high)
                if high - i > i - low:
                    if high > i:
                        stack[n] = Partition(i + 1, high)
                        n += 1
                    high = i - 1
                else:
                    if i > low:
                        stack[n] = Partition(low, i - 1)
                        n += 1
                    low = i + 1
            insertion_sort(A, R, low, high)
        return R
    if is_np_array:

        @wrap
        def run_quicksort(A):
            if A.ndim == 1:
                return run_quicksort1(A)
            else:
                for idx in np.ndindex(A.shape[:-1]):
                    run_quicksort1(A[idx])
                return A
    else:

        @wrap
        def run_quicksort(A):
            return run_quicksort1(A)

    @wrap
    def _run_quicksort(A):
        stack = [Partition(zero, zero)] * 100
        stack[0] = Partition(zero, len(A) - 1)
        n = 1
        while n > 0:
            n -= 1
            low, high = stack[n]
            while high - low >= SMALL_QUICKSORT:
                assert n < MAX_STACK
                l, r = partition3(A, low, high)
                if r == high:
                    high = l - 1
                elif l == low:
                    low = r + 1
                elif high - r > l - low:
                    stack[n] = Partition(r + 1, high)
                    n += 1
                    high = l - 1
                else:
                    stack[n] = Partition(low, l - 1)
                    n += 1
                    low = r + 1
            insertion_sort(A, low, high)
    return QuicksortImplementation(wrap, partition, partition3, insertion_sort, run_quicksort)