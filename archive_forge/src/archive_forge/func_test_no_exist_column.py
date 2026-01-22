import weakref
from unittest import mock
from cliff import show
from cliff.tests import base
def test_no_exist_column(self):
    test_show = ExerciseShowOne(mock.Mock(), [])
    parsed_args = mock.Mock()
    parsed_args.columns = ('no_exist_column',)
    parsed_args.formatter = 'test'
    with mock.patch.object(test_show, 'take_action') as mock_take_action:
        mock_take_action.return_value = (('Col1', 'Col2', 'Col3'), [])
        self.assertRaises(ValueError, test_show.run, parsed_args)