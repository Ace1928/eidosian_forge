def get_yaqlization_settings(class_or_object):
    return getattr(class_or_object, YAQLIZATION_ATTR, None)