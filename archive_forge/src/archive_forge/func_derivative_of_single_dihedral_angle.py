from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def derivative_of_single_dihedral_angle(self, tetIndex, i, j):
    """
        Gives the derivative of the single dihedral angle between face i and j
        of tetrahedron tetIndex with respect to the edge parameters.
        """
    s = len(self.mcomplex.Edges)
    result = vector(self.vertex_gram_matrices[0].base_ring(), s)
    tet = self.mcomplex.Tetrahedra[tetIndex]
    cofactor_matrices = _cofactor_matrices_for_submatrices(self.vertex_gram_matrices[tetIndex])
    vga = self.vertex_gram_adjoints[tetIndex]
    cii = vga[i, i]
    cij = vga[i, j]
    cjj = vga[j, j]
    dcij = -1 / sqrt(cii * cjj - cij ** 2)
    tmp = -dcij * cij / 2
    dcii = tmp / cii
    dcjj = tmp / cjj
    for length_edge, (m, n) in _OneSubsimplicesWithVertexIndices:
        l = tet.Class[length_edge].Index
        result[l] += dcij * _cofactor_derivative(cofactor_matrices, i, j, m, n) + dcii * _cofactor_derivative(cofactor_matrices, i, i, m, n) + dcjj * _cofactor_derivative(cofactor_matrices, j, j, m, n)
    return result