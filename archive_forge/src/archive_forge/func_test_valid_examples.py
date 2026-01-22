import unittest
from traits.observation._generated_parser import (
def test_valid_examples(self):
    good_examples = ['name', 'name123', 'name_a', '_name', 'foo.bar', 'foo  .  bar', 'foo:bar', 'foo  :  bar', 'foo,bar', 'foo  ,  bar', '[foo,bar,foo.spam]', '[foo, bar].baz', '[foo, [bar, baz]]:spam', 'foo:[bar.spam,baz]', 'foo.items', 'items', '+metadata_name']
    for good_example in good_examples:
        with self.subTest(good_example=good_example):
            try:
                PARSER.parse(good_example)
            except Exception:
                self.fail('Parsing {!r} is expected to succeed.'.format(good_example))