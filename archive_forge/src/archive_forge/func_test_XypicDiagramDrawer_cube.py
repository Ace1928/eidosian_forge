from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
from sympy.sets.sets import FiniteSet
def test_XypicDiagramDrawer_cube():
    A1 = Object('A1')
    A2 = Object('A2')
    A3 = Object('A3')
    A4 = Object('A4')
    A5 = Object('A5')
    A6 = Object('A6')
    A7 = Object('A7')
    A8 = Object('A8')
    f1 = NamedMorphism(A1, A2, 'f1')
    f2 = NamedMorphism(A1, A3, 'f2')
    f3 = NamedMorphism(A2, A4, 'f3')
    f4 = NamedMorphism(A3, A4, 'f3')
    f5 = NamedMorphism(A5, A6, 'f5')
    f6 = NamedMorphism(A5, A7, 'f6')
    f7 = NamedMorphism(A6, A8, 'f7')
    f8 = NamedMorphism(A7, A8, 'f8')
    f9 = NamedMorphism(A1, A5, 'f9')
    f10 = NamedMorphism(A2, A6, 'f10')
    f11 = NamedMorphism(A3, A7, 'f11')
    f12 = NamedMorphism(A4, A8, 'f11')
    d = Diagram([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
    grid = DiagramGrid(d)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\n& A_{5} \\ar[r]^{f_{5}} \\ar[ldd]_{f_{6}} & A_{6} \\ar[rdd]^{f_{7}} & \\\\\n& A_{1} \\ar[r]^{f_{1}} \\ar[d]^{f_{2}} \\ar[u]^{f_{9}} & A_{2} \\ar[d]^{f_{3}} \\ar[u]_{f_{10}} & \\\\\nA_{7} \\ar@/_3mm/[rrr]_{f_{8}} & A_{3} \\ar[r]^{f_{3}} \\ar[l]_{f_{11}} & A_{4} \\ar[r]^{f_{11}} & A_{8} \n}\n'
    grid = DiagramGrid(d, transpose=True)
    drawer = XypicDiagramDrawer()
    assert drawer.draw(d, grid) == '\\xymatrix{\n& & A_{7} \\ar@/^3mm/[ddd]^{f_{8}} \\\\\nA_{5} \\ar[d]_{f_{5}} \\ar[rru]^{f_{6}} & A_{1} \\ar[d]^{f_{1}} \\ar[r]^{f_{2}} \\ar[l]^{f_{9}} & A_{3} \\ar[d]_{f_{3}} \\ar[u]^{f_{11}} \\\\\nA_{6} \\ar[rrd]_{f_{7}} & A_{2} \\ar[r]^{f_{3}} \\ar[l]^{f_{10}} & A_{4} \\ar[d]_{f_{11}} \\\\\n& & A_{8} \n}\n'