important invariant is that the parts on the stack are themselves in
def enum_range(self, multiplicities, lb, ub):
    """Enumerate the partitions of a multiset with
        ``lb < num(parts) <= ub``.

        In particular, if partitions with exactly ``k`` parts are
        desired, call with ``(multiplicities, k - 1, k)``.  This
        method generalizes enum_all, enum_small, and enum_large.

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_range([2,2], 1, 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']]]

        """
    self.discarded = 0
    if ub <= 0 or lb >= sum(multiplicities):
        return
    self._initialize_enumeration(multiplicities)
    self.decrement_part_large(self.top_part(), 0, lb)
    while True:
        good_partition = True
        while self.spread_part_multiplicity():
            self.db_trace('spread 1')
            if not self.decrement_part_large(self.top_part(), 0, lb):
                self.db_trace('  Discarding (large cons)')
                self.discarded += 1
                good_partition = False
                break
            elif self.lpart >= ub:
                self.discarded += 1
                good_partition = False
                self.db_trace('  Discarding small cons')
                self.lpart = ub - 2
                break
        if good_partition:
            state = [self.f, self.lpart, self.pstack]
            yield state
        while not self.decrement_part_range(self.top_part(), lb, ub):
            self.db_trace('Failed decrement, going to backtrack')
            if self.lpart == 0:
                return
            self.lpart -= 1
            self.db_trace('Backtracked to')
        self.db_trace('decrement ok, about to expand')