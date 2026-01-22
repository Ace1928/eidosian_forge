import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_prettytable(self):

    class Struct(object):

        def __init__(self, **entries):
            self.__dict__.update(entries)
    columns = ['ID', 'Name']
    val = ['Name1', 'another', 'veeeery long']
    images = [Struct(**{'id': i ** 16, 'name': val[i]}) for i in range(len(val))]
    saved_stdout = sys.stdout
    try:
        sys.stdout = output_list = io.StringIO()
        utils.print_list(images, columns)
        sys.stdout = output_dict = io.StringIO()
        utils.print_dict({'K': 'k', 'Key': 'veeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeery long value'}, max_column_width=60)
    finally:
        sys.stdout = saved_stdout
    self.assertEqual('+-------+--------------+\n| ID    | Name         |\n+-------+--------------+\n|       | Name1        |\n| 1     | another      |\n| 65536 | veeeery long |\n+-------+--------------+\n', output_list.getvalue())
    self.assertEqual('+----------+--------------------------------------------------------------+\n| Property | Value                                                        |\n+----------+--------------------------------------------------------------+\n| K        | k                                                            |\n| Key      | veeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee |\n|          | eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee |\n|          | ery long value                                               |\n+----------+--------------------------------------------------------------+\n', output_dict.getvalue())