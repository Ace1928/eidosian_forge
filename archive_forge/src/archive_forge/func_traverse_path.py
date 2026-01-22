import os
def traverse_path(self, action, path, paths, checked):
    checked[path] = None
    for check_path, checker in paths:
        if path.startswith(check_path):
            action.check(check_path, checker)
            if not checker.inherit:
                break