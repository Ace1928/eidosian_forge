import json
def register_tag(cls):
    """
    Decorates a class to register it's json tag.
    """
    json_tags[TAG_PREFIX + getattr(cls, 'json_tag')] = cls
    return cls