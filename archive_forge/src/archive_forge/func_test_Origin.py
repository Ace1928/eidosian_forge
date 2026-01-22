def test_Origin():
    o1 = Origin('012345', 2, 4)
    o2 = Origin('012345', 4, 5)
    assert o1.caretize() == '012345\n  ^^'
    assert o2.caretize() == '012345\n    ^'
    o3 = Origin.combine([o1, o2])
    assert o3.code == '012345'
    assert o3.start == 2
    assert o3.end == 5
    assert o3.caretize(indent=2) == '  012345\n    ^^^'
    assert o3 == Origin('012345', 2, 5)

    class ObjWithOrigin(object):

        def __init__(self, origin=None):
            self.origin = origin
    o4 = Origin.combine([ObjWithOrigin(o1), ObjWithOrigin(), None])
    assert o4 == o1
    o5 = Origin.combine([ObjWithOrigin(o1), o2])
    assert o5 == o3
    assert Origin.combine([ObjWithOrigin(), ObjWithOrigin()]) is None
    from patsy.util import assert_no_pickling
    assert_no_pickling(Origin('', 0, 0))