from collections import defaultdict
def verify_non_overlapping(self, head_node, decref_blocks, entry):
    self.print('verify_non_overlapping'.center(80, '-'))
    todo = list(decref_blocks)
    while todo:
        cur_node = todo.pop()
        visited = set()
        workstack = [cur_node]
        del cur_node
        while workstack:
            cur_node = workstack.pop()
            self.print('cur_node', cur_node, '|', workstack)
            if cur_node in visited:
                continue
            if cur_node == entry:
                self.print('!! failed because we arrived at entry', cur_node)
                return False
            visited.add(cur_node)
            self.print(f'   {cur_node} preds {self.get_predecessors(cur_node)}')
            for pred in self.get_predecessors(cur_node):
                if pred in decref_blocks:
                    self.print('!! reject because predecessor in decref_blocks')
                    return False
                if pred != head_node:
                    workstack.append(pred)
    return True