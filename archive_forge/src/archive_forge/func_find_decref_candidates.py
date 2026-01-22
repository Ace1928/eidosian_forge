from collections import defaultdict
def find_decref_candidates(self, cur_node):
    self.print('find_decref_candidates'.center(80, '-'))
    path_stack = (cur_node,)
    found = False
    decref_blocks = set()
    for child in self.get_successors(cur_node):
        if not self.walk_child_for_decref(child, path_stack, decref_blocks):
            found = False
            break
        else:
            found = True
    if not found:
        return set()
    else:
        return decref_blocks