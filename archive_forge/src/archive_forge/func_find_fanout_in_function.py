from collections import defaultdict
def find_fanout_in_function(self):
    got = {}
    for cur_node in self.edges:
        for incref in (x for x in self.nodes[cur_node] if x == 'incref'):
            decref_blocks = self.find_fanout(cur_node)
            self.print('>>', cur_node, '===', decref_blocks)
            got[cur_node] = decref_blocks
    return got