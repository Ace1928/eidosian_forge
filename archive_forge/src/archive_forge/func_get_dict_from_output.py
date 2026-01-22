def get_dict_from_output(output):
    """Parse list of dictionaries, return a dictionary.

    :param output: list of dictionaries
    """
    obj = {}
    for item in output:
        obj[item['Property']] = str(item['Value'])
    return obj