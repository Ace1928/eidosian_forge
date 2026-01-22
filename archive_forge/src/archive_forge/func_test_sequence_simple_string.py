def test_sequence_simple_string(self):
    from boto.sdb.db.sequence import Sequence, increment_string
    s = Sequence(fnc=increment_string)
    self.sequences.append(s)
    assert s.val == 'A'
    assert s.next() == 'B'