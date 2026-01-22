from .error import *
from .nodes import *
import re
def descend_resolver(self, current_node, current_index):
    if not self.yaml_path_resolvers:
        return
    exact_paths = {}
    prefix_paths = []
    if current_node:
        depth = len(self.resolver_prefix_paths)
        for path, kind in self.resolver_prefix_paths[-1]:
            if self.check_resolver_prefix(depth, path, kind, current_node, current_index):
                if len(path) > depth:
                    prefix_paths.append((path, kind))
                else:
                    exact_paths[kind] = self.yaml_path_resolvers[path, kind]
    else:
        for path, kind in self.yaml_path_resolvers:
            if not path:
                exact_paths[kind] = self.yaml_path_resolvers[path, kind]
            else:
                prefix_paths.append((path, kind))
    self.resolver_exact_paths.append(exact_paths)
    self.resolver_prefix_paths.append(prefix_paths)