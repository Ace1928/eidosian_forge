def is_yaqlized(class_or_object):
    return hasattr(class_or_object, YAQLIZATION_ATTR)