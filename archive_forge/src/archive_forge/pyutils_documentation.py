import sys
Determine the 'parent' of a given dotted module name and (optional)
    member name.

    The idea is that ``getattr(parent_obj, final_attr)`` will equal
    get_named_object(module_name, member_name).

    :return: (module_name, member_name, final_attr) tuple.
    