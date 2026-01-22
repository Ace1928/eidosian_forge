important invariant is that the parts on the stack are themselves in
def decrement_part(self, part):
    """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        If you think of the v values in the part as a multi-digit
        integer (least significant digit on the right) this is
        basically decrementing that integer, but with the extra
        constraint that the leftmost digit cannot be decremented to 0.

        Parameters
        ==========

        part
           The part, represented as a list of PartComponent objects,
           which is to be decremented.

        """
    plen = len(part)
    for j in range(plen - 1, -1, -1):
        if j == 0 and part[j].v > 1 or (j > 0 and part[j].v > 0):
            part[j].v -= 1
            for k in range(j + 1, plen):
                part[k].v = part[k].u
            return True
    return False