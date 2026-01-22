important invariant is that the parts on the stack are themselves in
def decrement_part_large(self, part, amt, lb):
    """Decrements part, while respecting size constraint.

        A part can have no children which are of sufficient size (as
        indicated by ``lb``) unless that part has sufficient
        unallocated multiplicity.  When enforcing the size constraint,
        this method will decrement the part (if necessary) by an
        amount needed to ensure sufficient unallocated multiplicity.

        Returns True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        amt
            Can only take values 0 or 1.  A value of 1 means that the
            part must be decremented, and then the size constraint is
            enforced.  A value of 0 means just to enforce the ``lb``
            size constraint.

        lb
            The partitions produced by the calling enumeration must
            have more parts than this value.

        """
    if amt == 1:
        if not self.decrement_part(part):
            return False
    min_unalloc = lb - self.lpart
    if min_unalloc <= 0:
        return True
    total_mult = sum((pc.u for pc in part))
    total_alloc = sum((pc.v for pc in part))
    if total_mult <= min_unalloc:
        return False
    deficit = min_unalloc - (total_mult - total_alloc)
    if deficit <= 0:
        return True
    for i in range(len(part) - 1, -1, -1):
        if i == 0:
            if part[0].v > deficit:
                part[0].v -= deficit
                return True
            else:
                return False
        elif part[i].v >= deficit:
            part[i].v -= deficit
            return True
        else:
            deficit -= part[i].v
            part[i].v = 0