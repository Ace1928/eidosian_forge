import weakref
from unittest import mock
from cliff import show
from cliff.tests import base
def test_formatter_args(self):
    app = mock.Mock()
    test_show = ExerciseShowOne(app, [])
    parsed_args = mock.Mock()
    parsed_args.columns = ('Col1', 'Col2')
    parsed_args.formatter = 'test'
    test_show.run(parsed_args)
    f = test_show._formatter_plugins['test']
    self.assertEqual(1, len(f.args))
    args = f.args[0]
    self.assertEqual(list(parsed_args.columns), args[0])
    data = list(args[1])
    self.assertEqual([('a', 'A'), ('b', 'B')], data)