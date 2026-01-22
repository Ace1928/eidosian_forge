import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def print_group_type_list(gtypes):
    print_list(gtypes, ['ID', 'Name', 'Description'])