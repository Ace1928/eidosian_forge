from collections import defaultdict
def find_fanout(self, head_node):
    decref_blocks = self.find_decref_candidates(head_node)
    self.print('candidates', decref_blocks)
    if not decref_blocks:
        return None
    if not self.verify_non_overlapping(head_node, decref_blocks, entry=ENTRY):
        return None
    return set(decref_blocks)