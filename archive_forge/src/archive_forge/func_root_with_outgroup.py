import random
import sys
from . import Nodes
def root_with_outgroup(self, outgroup=None):
    """Define a tree's root with a reference group outgroup."""

    def _connect_subtree(parent, child):
        """Attach subtree starting with node child to parent (PRIVATE)."""
        for i, branch in enumerate(self.unrooted):
            if parent in branch[:2] and child in branch[:2]:
                branch = self.unrooted.pop(i)
                break
        else:
            raise TreeError('Unable to connect nodes for rooting: nodes %d and %d are not connected' % (parent, child))
        self.link(parent, child)
        self.node(child).data.branchlength = branch[2]
        self.node(child).data.support = branch[3]
        child_branches = [b for b in self.unrooted if child in b[:2]]
        for b in child_branches:
            if child == b[0]:
                succ = b[1]
            else:
                succ = b[0]
            _connect_subtree(child, succ)
    if outgroup is None:
        return self.root
    outgroup_node = self.is_monophyletic(outgroup)
    if outgroup_node == -1:
        return -1
    if len(self.node(self.root).succ) == 2 and outgroup_node in self.node(self.root).succ or outgroup_node == self.root:
        return self.root
    self.unroot()
    for i, b in enumerate(self.unrooted):
        if outgroup_node in b[:2] and self.node(outgroup_node).prev in b[:2]:
            root_branch = self.unrooted.pop(i)
            break
    else:
        raise TreeError('Unrooted and rooted Tree do not match')
    if outgroup_node == root_branch[1]:
        ingroup_node = root_branch[0]
    else:
        ingroup_node = root_branch[1]
    for n in self.all_ids():
        self.node(n).prev = None
        self.node(n).succ = []
    root = Nodes.Node(data=NodeData())
    self.add(root)
    self.root = root.id
    self.unrooted.append([root.id, ingroup_node, root_branch[2], root_branch[3]])
    self.unrooted.append([root.id, outgroup_node, 0.0, 0.0])
    _connect_subtree(root.id, ingroup_node)
    _connect_subtree(root.id, outgroup_node)
    oldroot = [i for i in self.all_ids() if self.node(i).prev is None and i != self.root]
    if len(oldroot) > 1:
        raise TreeError(f'Isolated nodes in tree description: {','.join(oldroot)}')
    elif len(oldroot) == 1:
        self.kill(oldroot[0])
    return self.root