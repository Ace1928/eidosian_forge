from pycadf import cadftype
def is_valid_action(value):
    for type in ACTION_TAXONOMY:
        if value.startswith(type):
            return True
    return False