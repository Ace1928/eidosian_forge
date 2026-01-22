from ..snap.t3mlite import simplex
from ..hyperboloid import *
def check_consistency_2(piece):
    tets = _find_all_tetrahedra(piece.tet)
    tets_set = set(tets)
    to_pieces_map = {}
    num_pieces = 0
    for tet in tets:
        for piece in tet.geodesic_pieces:
            num_pieces += 1
            if piece.tet is not tet:
                raise Exception('Piece.tet not pointing to tet.')
            if piece.next_.prev is not piece:
                raise Exception('Link list broken.')
            if piece.prev.next_ is not piece:
                raise Exception('Link list broken.')
            if piece.index != piece.next_.index:
                raise Exception('Index inconsistent.')
            if piece.index != piece.prev.index:
                raise Exception('Index inconsistent.')
            if piece.index not in to_pieces_map:
                l = flatten_link_list(piece)
                for i, p in enumerate(l):
                    if p is piece:
                        l == l[i:] + l[:i]
                        break
                else:
                    for i, p in enumerate(l):
                        if p.endpoints[0].subsimplex == simplex.T:
                            l == l[i:] + l[:i]
                            break
                to_pieces_map[piece.index] = l
    if False:
        for i, pieces in sorted(to_pieces_map.items()):
            print('Component %d (length %d):' % (i, len(pieces)))
            output_linked(pieces[0], tets_set)
        print('Total length: %d' % num_pieces)