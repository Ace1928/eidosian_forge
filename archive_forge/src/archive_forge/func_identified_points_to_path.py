from . import matrix
def identified_points_to_path(pt, path):
    return dict([(identified_pt, path) for identified_pt in point_identification_dict[pt]])