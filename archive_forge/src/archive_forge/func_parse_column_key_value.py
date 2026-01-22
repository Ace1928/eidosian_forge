import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
@staticmethod
def parse_column_key_value(table_schema, setting_string):
    """
        Parses 'setting_string' as str formatted in <column>[:<key>]=<value>
        and returns str type 'column' and json formatted 'value'
        """
    if ':' in setting_string:
        column, value = setting_string.split(':', 1)
    elif '=' in setting_string:
        column, value = setting_string.split('=', 1)
    else:
        column = setting_string
        value = None
    if value is not None:
        type_ = table_schema.columns[column].type
        value = datum_from_string(type_, value)
    return (column, value)