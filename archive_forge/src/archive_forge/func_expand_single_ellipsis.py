from collections import OrderedDict
def expand_single_ellipsis(numel_pre_glob, numel_post_glob, names):
    return names[numel_pre_glob:len(names) - numel_post_glob]