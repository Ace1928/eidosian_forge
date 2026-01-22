import pytest
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
import cirq
def test_supports_arbitrary_offsets():
    m = 1 << 60
    q_neg = BucketPriorityQueue()
    q_neg.enqueue(-m + 0, 'b')
    q_neg.enqueue(-m - 4, 'a')
    q_neg.enqueue(-m + 4, 'c')
    assert list(q_neg) == [(-m - 4, 'a'), (-m, 'b'), (-m + 4, 'c')]
    q_pos = BucketPriorityQueue()
    q_pos.enqueue(m + 0, 'b')
    q_pos.enqueue(m + 4, 'c')
    q_pos.enqueue(m - 4, 'a')
    assert list(q_pos) == [(m - 4, 'a'), (m, 'b'), (m + 4, 'c')]