import pytest
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
import cirq
def test_drop_duplicates_dequeue():
    q0 = BucketPriorityQueue()
    q1 = BucketPriorityQueue(drop_duplicate_entries=False)
    q2 = BucketPriorityQueue(drop_duplicate_entries=True)
    for q in [q0, q1, q2]:
        q.enqueue(0, 'a')
        q.enqueue(0, 'b')
        q.enqueue(0, 'a')
        q.dequeue()
        q.enqueue(0, 'b')
        q.enqueue(0, 'a')
    assert q0 == q1 == BucketPriorityQueue([(0, 'b'), (0, 'a'), (0, 'b'), (0, 'a')])
    assert q2 == BucketPriorityQueue([(0, 'b'), (0, 'a')], drop_duplicate_entries=True)