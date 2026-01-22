def json_index_face_set(t):
    from itertools import chain

    def flatten(ll):
        return list(chain.from_iterable(ll))
    return FaceGeometry(vertices=flatten(t['vertices']), face3=flatten(t['face3']), face4=flatten(t['face4']), facen=t['facen'])