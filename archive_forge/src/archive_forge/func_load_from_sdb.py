import os
import re
import warnings
import boto
from boto.compat import expanduser, ConfigParser, NoOptionError, NoSectionError, StringIO
def load_from_sdb(self, domain_name, item_name):
    from boto.compat import json
    sdb = boto.connect_sdb()
    domain = sdb.lookup(domain_name)
    item = domain.get_item(item_name)
    for section in item.keys():
        if not self.has_section(section):
            self.add_section(section)
        d = json.loads(item[section])
        for attr_name in d.keys():
            attr_value = d[attr_name]
            if attr_value is None:
                attr_value = 'None'
            if isinstance(attr_value, bool):
                self.setbool(section, attr_name, attr_value)
            else:
                self.set(section, attr_name, attr_value)