import unittest
import simplejson as json
def test_for_json_ignored_if_not_true_with_dict_subclass(self):
    for for_json in (None, False):
        self.assertRoundTrip(DictForJson(a=1), {'a': 1}, for_json=for_json)