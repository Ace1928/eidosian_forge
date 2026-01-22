import unittest
from traits.util.weakiddict import WeakIDDict, WeakIDKeyDict
def test_weak_keys_values(self):
    wd = WeakIDDict()
    keep = []
    dont_keep = []
    values = list(map(WeakreffableInt, range(10)))
    for n, i in enumerate(values, 1):
        key = AllTheSame()
        if not i.value % 2:
            keep.append(key)
        else:
            dont_keep.append(key)
        wd[key] = i
        del key
        self.assertEqual(len(wd), n)
    self.assertEqual(len(wd), 10)
    del dont_keep
    self.assertEqual(len(wd), 5)
    self.assertCountEqual(list(wd.values()), list(map(WeakreffableInt, [0, 2, 4, 6, 8])))
    self.assertEqual([wd[k] for k in keep], list(map(WeakreffableInt, [0, 2, 4, 6, 8])))
    self.assertCountEqual([id(k) for k in wd.keys()], [id(k) for k in wd])
    self.assertCountEqual([id(k) for k in wd.keys()], [id(k) for k in keep])
    del values[0:2]
    self.assertEqual(len(wd), 4)
    del values[0:2]
    self.assertEqual(len(wd), 3)
    del values[0:2]
    self.assertEqual(len(wd), 2)
    del values[0:2]
    self.assertEqual(len(wd), 1)
    del values[0:2]
    self.assertEqual(len(wd), 0)