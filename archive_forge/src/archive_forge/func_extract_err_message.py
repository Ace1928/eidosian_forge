import six
def extract_err_message(exception):
    if exception.args:
        return exception.args[0]
    else:
        return None