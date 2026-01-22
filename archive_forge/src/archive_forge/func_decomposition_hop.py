from __future__ import annotations
import numpy as np
from .polytopes import ConvexPolytopeData, PolytopeData, manual_get_vertex, polytope_has_element
def decomposition_hop(target_coordinate, strengths):
    """
    Given a `target_coordinate` and a list of interaction `strengths`, produces a new canonical
    coordinate which is one step back along `strengths`.

    `target_coordinate` is taken to be in positive canonical coordinates, and the entries of
    strengths are taken to be [0, pi], so that (sj / 2, 0, 0) is a positive canonical coordinate.
    """
    target_coordinate = [x / (np.pi / 2) for x in target_coordinate]
    strengths = [x / np.pi for x in strengths]
    augmented_coordinate = get_augmented_coordinate(target_coordinate, strengths)
    specialized_polytope = None
    for cp in xx_region_polytope.convex_subpolytopes:
        if not polytope_has_element(cp, augmented_coordinate):
            continue
        if 'AF=B1' in cp.name:
            af, _, _ = target_coordinate
        elif 'AF=B2' in cp.name:
            _, af, _ = target_coordinate
        elif 'AF=B3' in cp.name:
            _, _, af = target_coordinate
        else:
            raise ValueError("Couldn't find a coordinate to fix.")
        raw_convex_polytope = next((cpp for cpp in xx_lift_polytope.convex_subpolytopes if cpp.name == cp.name), None)
        coefficient_dict = {}
        for inequality in raw_convex_polytope.inequalities:
            if inequality[1] == 0 and inequality[2] == 0:
                continue
            offset = inequality[0] + inequality[3] * af + inequality[4] * augmented_coordinate[0] + inequality[5] * augmented_coordinate[1] + inequality[6] * augmented_coordinate[2] + inequality[7] * augmented_coordinate[3] + inequality[8] * augmented_coordinate[4] + inequality[9] * augmented_coordinate[5] + inequality[10] * augmented_coordinate[6]
            if offset <= coefficient_dict.get((inequality[1], inequality[2]), offset):
                coefficient_dict[inequality[1], inequality[2]] = offset
        specialized_polytope = PolytopeData(convex_subpolytopes=[ConvexPolytopeData(inequalities=[[v, h, l] for (h, l), v in coefficient_dict.items()])])
        break
    if specialized_polytope is None:
        raise ValueError('Failed to match a constrained_polytope summand.')
    ah, al = manual_get_vertex(specialized_polytope)
    return [x * (np.pi / 2) for x in sorted([ah, al, af], reverse=True)]