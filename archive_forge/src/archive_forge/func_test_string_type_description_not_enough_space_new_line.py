from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import custom_descriptions
from fire import testutils
def test_string_type_description_not_enough_space_new_line(self):
    component = 'Lorem ipsum dolor sit amet'
    description = custom_descriptions.GetDescription(obj=component, available_space=10, line_length=LINE_LENGTH)
    self.assertEqual(description, 'The string "Lorem ipsum dolor sit amet"')