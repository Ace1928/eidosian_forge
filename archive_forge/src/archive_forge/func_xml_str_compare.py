import os
import uuid
import xmltodict
from pytest import skip, fixture
from mock import patch
def xml_str_compare(first, second):
    first_dict = xmltodict.parse(first)
    second_dict = xmltodict.parse(second)
    sort_dict(first_dict)
    sort_dict(second_dict)
    return first_dict == second_dict