import re
import sys
from optparse import OptionParser
from rdkit import Chem
def cansmirk(lhs, rhs, context):
    isotope_track = {}
    stars = lhs.count('*')
    if stars > 1:
        lhs_sym = get_symmetry_class(lhs)
        rhs_sym = get_symmetry_class(rhs)
    if stars == 2:
        if lhs_sym[0] != lhs_sym[1] and rhs_sym[0] != rhs_sym[1]:
            isotope_track = build_track_dictionary(lhs, stars)
            lhs = switch_labels_on_position(lhs)
            rhs = switch_labels(isotope_track, stars, rhs)
            context = switch_labels(isotope_track, stars, context)
        elif lhs_sym[0] == lhs_sym[1] and rhs_sym[0] == rhs_sym[1]:
            lhs = switch_labels_on_position(lhs)
            rhs = switch_labels_on_position(rhs)
        elif lhs_sym[0] == lhs_sym[1] and rhs_sym[0] != rhs_sym[1]:
            lhs = switch_labels_on_position(lhs)
            isotope_track = build_track_dictionary(rhs, stars)
            rhs = switch_labels_on_position(rhs)
            context = switch_labels(isotope_track, stars, context)
        elif lhs_sym[0] != lhs_sym[1] and rhs_sym[0] == rhs_sym[1]:
            isotope_track = build_track_dictionary(lhs, stars)
            lhs = switch_labels_on_position(lhs)
            context = switch_labels(isotope_track, stars, context)
            rhs = switch_labels_on_position(rhs)
    elif stars == 3:
        if (lhs_sym[0] == lhs_sym[1] and lhs_sym[1] == lhs_sym[2] and (lhs_sym[0] == lhs_sym[2])) and (rhs_sym[0] == rhs_sym[1] and rhs_sym[1] == rhs_sym[2] and (rhs_sym[0] == rhs_sym[2])):
            lhs = switch_labels_on_position(lhs)
            rhs = switch_labels_on_position(rhs)
        elif (lhs_sym[0] == lhs_sym[1] and lhs_sym[1] == lhs_sym[2] and (lhs_sym[0] == lhs_sym[2])) and (rhs_sym[0] != rhs_sym[1] and rhs_sym[1] != rhs_sym[2] and (rhs_sym[0] != rhs_sym[2])):
            lhs = switch_labels_on_position(lhs)
            isotope_track = build_track_dictionary(rhs, stars)
            rhs = switch_labels_on_position(rhs)
            context = switch_labels(isotope_track, stars, context)
        elif (lhs_sym[0] != lhs_sym[1] and lhs_sym[1] != lhs_sym[2] and (lhs_sym[0] != lhs_sym[2])) and (rhs_sym[0] != rhs_sym[1] and rhs_sym[1] != rhs_sym[2] and (rhs_sym[0] != rhs_sym[2])):
            isotope_track = build_track_dictionary(lhs, stars)
            lhs = switch_labels_on_position(lhs)
            rhs = switch_labels(isotope_track, stars, rhs)
            context = switch_labels(isotope_track, stars, context)
        elif (lhs_sym[0] != lhs_sym[1] and lhs_sym[1] != lhs_sym[2] and (lhs_sym[0] != lhs_sym[2])) and (rhs_sym[0] == rhs_sym[1] and rhs_sym[1] == rhs_sym[2] and (rhs_sym[0] == rhs_sym[2])):
            isotope_track = build_track_dictionary(lhs, stars)
            lhs = switch_labels_on_position(lhs)
            context = switch_labels(isotope_track, stars, context)
            rhs = switch_labels_on_position(rhs)
        elif lhs_sym[0] != lhs_sym[1] and lhs_sym[1] != lhs_sym[2] and (lhs_sym[0] != lhs_sym[2]):
            isotope_track = build_track_dictionary(lhs, stars)
            lhs = switch_labels_on_position(lhs)
            rhs = switch_labels(isotope_track, stars, rhs)
            context = switch_labels(isotope_track, stars, context)
            if rhs_sym[0] == rhs_sym[1]:
                rhs = switch_specific_labels_on_symmetry(rhs, rhs_sym, 1, 2)
            elif rhs_sym[1] == rhs_sym[2]:
                rhs = switch_specific_labels_on_symmetry(rhs, rhs_sym, 2, 3)
            elif rhs_sym[0] == rhs_sym[2]:
                rhs = switch_specific_labels_on_symmetry(rhs, rhs_sym, 1, 3)
        elif lhs_sym[0] == lhs_sym[1] and lhs_sym[1] == lhs_sym[2] and (lhs_sym[0] == lhs_sym[2]):
            lhs = switch_labels_on_position(lhs)
            isotope_track = build_track_dictionary(rhs, stars)
            rhs = switch_labels_on_position(rhs)
            context = switch_labels(isotope_track, stars, context)
        else:
            isotope_track = build_track_dictionary(lhs, stars)
            lhs = switch_labels_on_position(lhs)
            rhs = switch_labels(isotope_track, stars, rhs)
            context = switch_labels(isotope_track, stars, context)
            if lhs_sym[0] == lhs_sym[1]:
                rhs = switch_specific_labels_on_symmetry(rhs, rhs_sym, 1, 2)
            elif lhs_sym[1] == lhs_sym[2]:
                rhs = switch_specific_labels_on_symmetry(rhs, rhs_sym, 2, 3)
            elif lhs_sym[0] == lhs_sym[2]:
                rhs = switch_specific_labels_on_symmetry(rhs, rhs_sym, 1, 3)
    smirk = '%s>>%s' % (lhs, rhs)
    return (smirk, context)