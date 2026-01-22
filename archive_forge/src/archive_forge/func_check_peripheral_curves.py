from ..snap.t3mlite import simplex
from ..hyperboloid import *
def check_peripheral_curves(tets):
    for tet in tets:
        for f in simplex.TwoSubsimplices:
            neighbor = tet.Neighbor[f]
            gluing = tet.Gluing[f]
            other_f = gluing.image(f)
            sgn = gluing.sign()
            for v in simplex.ZeroSubsimplices:
                v_comp = simplex.comp(v)
                other_v = gluing.image(v)
                for ml in range(2):
                    for sheet_index in range(2):
                        sheet = tet.PeripheralCurves[ml][sheet_index][v]
                        if f == v_comp:
                            if sum(sheet.values()) != 0:
                                raise Exception('Not adding up to zero. %r' % tet)
                            if sheet[v_comp] != 0:
                                raise Exception('Diagonal entry for peripheral curve.')
                        else:
                            if sgn == 0:
                                other_sheet_index = 1 - sheet_index
                            else:
                                other_sheet_index = sheet_index
                            a = sheet[f]
                            b = neighbor.PeripheralCurves[ml][other_sheet_index][other_v][other_f]
                            if a + b != 0:
                                raise Exception('Peripheral curve not adding up.')