def test_na_writable_attributes_deletion():
    a = np.NA(2)
    attr = ['payload', 'dtype']
    for s in attr:
        assert_raises(AttributeError, delattr, a, s)