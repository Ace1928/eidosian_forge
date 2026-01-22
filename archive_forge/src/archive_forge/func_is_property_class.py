import re
def is_property_class(class_):
    return _mf2_properties_re.match(class_)