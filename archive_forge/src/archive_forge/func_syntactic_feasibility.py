import sys
def syntactic_feasibility(self, G1_node, G2_node):
    """Returns True if adding (G1_node, G2_node) is syntactically feasible.

        This function returns True if it is adding the candidate pair
        to the current partial isomorphism/monomorphism mapping is allowable.
        The addition is allowable if the inclusion of the candidate pair does
        not make it impossible for an isomorphism/monomorphism to be found.
        """
    if self.test == 'mono':
        if self.G1.number_of_edges(G1_node, G1_node) < self.G2.number_of_edges(G2_node, G2_node):
            return False
    elif self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(G2_node, G2_node):
        return False
    if self.test != 'mono':
        for predecessor in self.G1.pred[G1_node]:
            if predecessor in self.core_1:
                if self.core_1[predecessor] not in self.G2.pred[G2_node]:
                    return False
                elif self.G1.number_of_edges(predecessor, G1_node) != self.G2.number_of_edges(self.core_1[predecessor], G2_node):
                    return False
    for predecessor in self.G2.pred[G2_node]:
        if predecessor in self.core_2:
            if self.core_2[predecessor] not in self.G1.pred[G1_node]:
                return False
            elif self.test == 'mono':
                if self.G1.number_of_edges(self.core_2[predecessor], G1_node) < self.G2.number_of_edges(predecessor, G2_node):
                    return False
            elif self.G1.number_of_edges(self.core_2[predecessor], G1_node) != self.G2.number_of_edges(predecessor, G2_node):
                return False
    if self.test != 'mono':
        for successor in self.G1[G1_node]:
            if successor in self.core_1:
                if self.core_1[successor] not in self.G2[G2_node]:
                    return False
                elif self.G1.number_of_edges(G1_node, successor) != self.G2.number_of_edges(G2_node, self.core_1[successor]):
                    return False
    for successor in self.G2[G2_node]:
        if successor in self.core_2:
            if self.core_2[successor] not in self.G1[G1_node]:
                return False
            elif self.test == 'mono':
                if self.G1.number_of_edges(G1_node, self.core_2[successor]) < self.G2.number_of_edges(G2_node, successor):
                    return False
            elif self.G1.number_of_edges(G1_node, self.core_2[successor]) != self.G2.number_of_edges(G2_node, successor):
                return False
    if self.test != 'mono':
        num1 = 0
        for predecessor in self.G1.pred[G1_node]:
            if predecessor in self.in_1 and predecessor not in self.core_1:
                num1 += 1
        num2 = 0
        for predecessor in self.G2.pred[G2_node]:
            if predecessor in self.in_2 and predecessor not in self.core_2:
                num2 += 1
        if self.test == 'graph':
            if num1 != num2:
                return False
        elif not num1 >= num2:
            return False
        num1 = 0
        for successor in self.G1[G1_node]:
            if successor in self.in_1 and successor not in self.core_1:
                num1 += 1
        num2 = 0
        for successor in self.G2[G2_node]:
            if successor in self.in_2 and successor not in self.core_2:
                num2 += 1
        if self.test == 'graph':
            if num1 != num2:
                return False
        elif not num1 >= num2:
            return False
        num1 = 0
        for predecessor in self.G1.pred[G1_node]:
            if predecessor in self.out_1 and predecessor not in self.core_1:
                num1 += 1
        num2 = 0
        for predecessor in self.G2.pred[G2_node]:
            if predecessor in self.out_2 and predecessor not in self.core_2:
                num2 += 1
        if self.test == 'graph':
            if num1 != num2:
                return False
        elif not num1 >= num2:
            return False
        num1 = 0
        for successor in self.G1[G1_node]:
            if successor in self.out_1 and successor not in self.core_1:
                num1 += 1
        num2 = 0
        for successor in self.G2[G2_node]:
            if successor in self.out_2 and successor not in self.core_2:
                num2 += 1
        if self.test == 'graph':
            if num1 != num2:
                return False
        elif not num1 >= num2:
            return False
        num1 = 0
        for predecessor in self.G1.pred[G1_node]:
            if predecessor not in self.in_1 and predecessor not in self.out_1:
                num1 += 1
        num2 = 0
        for predecessor in self.G2.pred[G2_node]:
            if predecessor not in self.in_2 and predecessor not in self.out_2:
                num2 += 1
        if self.test == 'graph':
            if num1 != num2:
                return False
        elif not num1 >= num2:
            return False
        num1 = 0
        for successor in self.G1[G1_node]:
            if successor not in self.in_1 and successor not in self.out_1:
                num1 += 1
        num2 = 0
        for successor in self.G2[G2_node]:
            if successor not in self.in_2 and successor not in self.out_2:
                num2 += 1
        if self.test == 'graph':
            if num1 != num2:
                return False
        elif not num1 >= num2:
            return False
    return True