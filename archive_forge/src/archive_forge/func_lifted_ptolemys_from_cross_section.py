from ...snap import t3mlite as t3m
def lifted_ptolemys_from_cross_section(cusp_cross_section, lifted_one_cocycle=None):
    """
    Given a cusp cross section, compute lifted Ptolemy coordinates
    (i.e., logarithms of the Ptolemy coordinates) returned as a dictionary
    (e.g., the key for the Ptolemy coordinate for the edge from
    vertex 0 to vertex 3 or simplex 4 is c_1001_4).

    For complete cusp cross sections (where no lifted_one_cocycle is
    necessary), we use Zickert's algorithm (Christian Zickert, The
    volume and Chern-Simons invariant of a representation, Duke
    Math. J. 150 no. 3 (2009) 489-532, math.GT/0710.2049). In this
    case, all values for keys corresponding to the same edge in the
    triangulation are guaranteed to be the same.

    For the incomplete cusp cross sections, a lifted_one_cocycle
    needs to be given. This cocycle is a lift of the cocycle one_cocycle
    given to ComplexCuspCrossSection.fromManifoldAndShapes.
    More precisely, lifted_one_cocycle is in C^1(boundary M; C) and
    needs to map to one_cocycle in C^1(boundary M; C^*).
    """
    result = {}
    some_tet = cusp_cross_section.mcomplex.Tetrahedra[0]
    some_z = some_tet.ShapeParameters[t3m.E01]
    CIF = some_z.parent()
    for edge in cusp_cross_section.mcomplex.Edges:
        for i, (tet, perm) in enumerate(edge.embeddings()):
            v0 = perm.image(t3m.V0)
            v1 = perm.image(t3m.V1)
            v2 = perm.image(t3m.V2)
            e = v0 | v1
            face = e | v2
            if i == 0:
                l1 = CIF(tet.horotriangles[v0].lengths[face])
                l2 = CIF(tet.horotriangles[v1].lengths[face])
                ptolemy = 1 / (l1 * l2).sqrt()
                ptolemy = ptolemy.log()
            elif lifted_one_cocycle:
                ptolemy -= lifted_one_cocycle[tet.Index, face, v0]
                ptolemy -= lifted_one_cocycle[tet.Index, face, v1]
            result[_ptolemy_coordinate_key(tet.Index, e)] = ptolemy
    return result