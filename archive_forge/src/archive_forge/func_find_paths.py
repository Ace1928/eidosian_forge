import re
from . import schema
from .search import Search
from .resources import CObject, Project, Projects, Experiment, Experiments
from .uriutil import inv_translate_uri
from .errors import ProgrammingError
def find_paths(element, path=[]):
    resources_dict = schema.resources_tree
    element = element.strip('/')
    paths = []
    if path == []:
        path = [element]
    init_path = path[:]
    for key in resources_dict.keys():
        path = init_path[:]
        if element in resources_dict[key]:
            path.append(key)
            look_again = find_paths(key, path)
            if look_again != []:
                paths.extend(look_again)
            else:
                path.reverse()
                paths.append('/' + '/'.join(path))
    return paths