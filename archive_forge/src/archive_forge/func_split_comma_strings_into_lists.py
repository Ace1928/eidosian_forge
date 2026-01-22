from __future__ import (absolute_import, division, print_function)
@staticmethod
def split_comma_strings_into_lists(obj):
    """
        Splits a CSV String into a list.  Also takes a dictionary, and converts any CSV strings in any key, to a list.

        :param obj: object in CSV format to be parsed.
        :type obj: str or dict

        :return: A list containing the CSV items.
        :rtype: list
        """
    return_obj = ()
    if isinstance(obj, dict):
        if len(obj) > 0:
            for k, v in obj.items():
                if isinstance(v, str):
                    new_list = list()
                    if ',' in v:
                        new_items = v.split(',')
                        for item in new_items:
                            new_list.append(item.strip())
                        obj[k] = new_list
            return_obj = obj
    elif isinstance(obj, str):
        return_obj = obj.replace(' ', '').split(',')
    return return_obj