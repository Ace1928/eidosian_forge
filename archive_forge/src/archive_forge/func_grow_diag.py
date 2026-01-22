from collections import defaultdict
def grow_diag():
    """
        Search for the neighbor points and them to the intersected alignment
        points if criteria are met.
        """
    prev_len = len(alignment) - 1
    while prev_len < len(alignment):
        no_new_points = True
        for e in range(srclen):
            for f in range(trglen):
                if (e, f) in alignment:
                    for neighbor in neighbors:
                        neighbor = tuple((i + j for i, j in zip((e, f), neighbor)))
                        e_new, f_new = neighbor
                        if (e_new not in aligned and f_new not in aligned) and neighbor in union:
                            alignment.add(neighbor)
                            aligned['e'].add(e_new)
                            aligned['f'].add(f_new)
                            prev_len += 1
                            no_new_points = False
        if no_new_points:
            break