import cvxpy.lin_ops.lin_op as lo
@staticmethod
def get_var_offsets(variables):
    var_shapes = {}
    var_offsets = {}
    id_map = {}
    vert_offset = 0
    for x in variables:
        var_shapes[x.id] = x.shape
        var_offsets[x.id] = vert_offset
        id_map[x.id] = (vert_offset, x.size)
        vert_offset += x.size
    return (id_map, var_offsets, vert_offset, var_shapes)