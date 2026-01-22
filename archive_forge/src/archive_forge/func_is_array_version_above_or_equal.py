from __future__ import absolute_import, division, print_function
import datetime
import uuid
def is_array_version_above_or_equal(array_obj_client, arr_version_to_check):
    if arr_version_to_check is None:
        return False
    resp = array_obj_client.get()
    if resp is None:
        return False
    array_version = resp.attrs.get('version')
    if array_version is None:
        return False
    temp = array_version.split('-')
    array_version = temp[0]
    arr_version = array_version.split('.')
    version_to_check = arr_version_to_check.split('.')
    if arr_version[0] > version_to_check[0]:
        return True
    elif arr_version[0] >= version_to_check[0] and arr_version[1] >= version_to_check[1]:
        return True
    elif arr_version[0] >= version_to_check[0] and arr_version[1] >= version_to_check[1] and (arr_version[2] >= version_to_check[2]):
        return True
    return False