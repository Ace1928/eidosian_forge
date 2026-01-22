def test_sequence_string(self):
    """Test the String incrementation sequence"""
    from boto.sdb.db.sequence import Sequence, increment_string
    s = Sequence(fnc=increment_string)
    self.sequences.append(s)
    assert s.val == 'A'
    assert s.next() == 'B'
    s.val = 'Z'
    assert s.val == 'Z'
    assert s.next() == 'AA'