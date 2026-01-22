from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
from sympy.sets.sets import FiniteSet
def test_XypicDiagramDrawer_triangle():
    A = Object('A')
    B = Object('B')
    C = Object('C')
    f = NamedMorphism(A, B, 'f')
    g = NamedMorphism(B, C, 'g')
    d = Diagram([f, g], {g * f: 'unique'})
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[d]_{g\\circ f} \\ar[r]^{f} & B \\ar[ld]^{g} \\\\\nC & \n}\n'
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar[r]^{g\\circ f} \\ar[d]_{f} & C \\\\\nB \\ar[ru]_{g} & \n}\n'
    assert drawer.draw(d, grid, masked=[g]) == '\\xymatrix{\nA \\ar[r]^{g\\circ f} \\ar[d]_{f} & C \\\\\nB & \n}\n'

    def formatter(astr):
        astr.label = '\\exists !' + astr.label
        astr.arrow_style = '{-->}'
    drawer.arrow_formatters['unique'] = formatter
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar@{-->}[r]^{\\exists !g\\circ f} \\ar[d]_{f} & C \\\\\nB \\ar[ru]_{g} & \n}\n'

    def default_formatter(astr):
        astr.label_displacement = '(0.45)'
    drawer.default_arrow_formatter = default_formatter
    assert drawer.draw(d, grid) == '\\xymatrix{\nA \\ar@{-->}[r]^(0.45){\\exists !g\\circ f} \\ar[d]_(0.45){f} & C \\\\\nB \\ar[ru]_(0.45){g} & \n}\n'
    f1 = NamedMorphism(B, A, 'f1')
    f2 = NamedMorphism(A, B, 'f2')
    g1 = NamedMorphism(C, B, 'g1')
    g2 = NamedMorphism(B, C, 'g2')
    d = Diagram([f, f1, f2, g, g1, g2], {f1 * g1: 'unique', g2 * f2: 'unique'})
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid, masked=[f1 * g1 * g2 * f2, g2 * f2 * f1 * g1]) == '\\xymatrix{\nA \\ar[r]^{g_{2}\\circ f_{2}} \\ar[d]_{f} \\ar@/^3mm/[d]^{f_{2}} & C \\ar@/^3mm/[l]^{f_{1}\\circ g_{1}} \\ar@/^3mm/[ld]^{g_{1}} \\\\\nB \\ar@/^3mm/[u]^{f_{1}} \\ar[ru]_{g} \\ar@/^3mm/[ru]^{g_{2}} & \n}\n'