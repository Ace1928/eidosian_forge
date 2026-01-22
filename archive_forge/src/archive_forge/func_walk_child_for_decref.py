from collections import defaultdict
def walk_child_for_decref(self, cur_node, path_stack, decref_blocks, depth=10):
    indent = ' ' * len(path_stack)
    self.print(indent, 'walk', path_stack, cur_node)
    if depth <= 0:
        return False
    if cur_node in path_stack:
        if cur_node == path_stack[0]:
            return False
        return True
    if self.has_decref(cur_node):
        decref_blocks.add(cur_node)
        self.print(indent, 'found decref')
        return True
    depth -= 1
    path_stack += (cur_node,)
    found = False
    for child in self.get_successors(cur_node):
        if not self.walk_child_for_decref(child, path_stack, decref_blocks):
            found = False
            break
        else:
            found = True
    self.print(indent, f'ret {found}')
    return found