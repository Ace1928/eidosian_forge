from collections import OrderedDict
def suggestion_list(inp, options):
    """
     Given an invalid input string and a list of valid options, returns a filtered
     list of valid options sorted based on their similarity with the input.
    """
    options_by_distance = OrderedDict()
    input_threshold = len(inp) / 2
    for option in options:
        distance = lexical_distance(inp, option)
        threshold = max(input_threshold, len(option) / 2, 1)
        if distance <= threshold:
            options_by_distance[option] = distance
    return sorted(list(options_by_distance.keys()), key=lambda k: options_by_distance[k])