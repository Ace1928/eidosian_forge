import os
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions as tempest_exc
def show_to_dict(self, output):
    obj = {}
    items = self.parser.listing(output)
    for item in items:
        obj[item['Field']] = str(item['Value'])
    return dict(((self._key_name(k), v) for k, v in obj.items()))