import numpy as np
def leaf_index_tree(self, X: np.ndarray, tree_id: int) -> int:
    """Computes the leaf index for one tree."""
    index = self.root_index[tree_id]
    while self.atts.nodes_modes[index] != 'LEAF':
        x = X[self.atts.nodes_featureids[index]]
        if np.isnan(x):
            r = self.atts.nodes_missing_value_tracks_true[index] >= 1
        else:
            rule = self.atts.nodes_modes[index]
            th = self.atts.nodes_values[index]
            if rule == 'BRANCH_LEQ':
                r = x <= th
            elif rule == 'BRANCH_LT':
                r = x < th
            elif rule == 'BRANCH_GTE':
                r = x >= th
            elif rule == 'BRANCH_GT':
                r = x > th
            elif rule == 'BRANCH_EQ':
                r = x == th
            elif rule == 'BRANCH_NEQ':
                r = x != th
            else:
                raise ValueError(f'Unexpected rule {rule!r} for node index {index}.')
        nid = self.atts.nodes_truenodeids[index] if r else self.atts.nodes_falsenodeids[index]
        index = self.node_index[tree_id, nid]
    return index