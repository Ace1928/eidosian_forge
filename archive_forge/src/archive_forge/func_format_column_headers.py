import logging
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient import exceptions
def format_column_headers(columns):
    column_headers = []
    for column in columns:
        column_headers.append(column.replace('_', ' ').title().replace('Id', 'ID'))
    return column_headers