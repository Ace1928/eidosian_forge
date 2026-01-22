import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_contrib_box_nms')
def convert_contrib_box_nms(node, **kwargs):
    """Map MXNet's _contrib_box_nms operator to ONNX
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')
    overlap_thresh = float(attrs.get('overlap_thresh', '0.5'))
    valid_thresh = float(attrs.get('valid_thresh', '0'))
    topk = int(attrs.get('topk', '-1'))
    coord_start = int(attrs.get('coord_start', '2'))
    score_index = int(attrs.get('score_index', '1'))
    id_index = int(attrs.get('id_index', '-1'))
    force_suppress = attrs.get('force_suppress', 'True')
    background_id = int(attrs.get('background_id', '-1'))
    in_format = attrs.get('in_format', 'corner')
    out_format = attrs.get('out_format', 'corner')
    center_point_box = 0 if in_format == 'corner' else 1
    if topk == -1:
        topk = 2 ** 31 - 1
    if in_format != out_format:
        raise NotImplementedError('box_nms does not currently support in_fomat != out_format')
    if background_id != -1:
        raise NotImplementedError('box_nms does not currently support background_id != -1')
    if id_index != -1 or force_suppress == 'False':
        logging.warning('box_nms: id_idex != -1 or/and force_suppress == False detected. However, due to ONNX limitations, boxes of different categories will NOT be exempted from suppression. This might lead to different behavior than native MXNet')
    create_tensor([coord_start], name + '_cs', kwargs['initializer'])
    create_tensor([coord_start + 4], name + '_cs_p4', kwargs['initializer'])
    create_tensor([score_index], name + '_si', kwargs['initializer'])
    create_tensor([score_index + 1], name + '_si_p1', kwargs['initializer'])
    create_tensor([topk], name + '_topk', kwargs['initializer'])
    create_tensor([overlap_thresh], name + '_ot', kwargs['initializer'], dtype=np.float32)
    create_tensor([valid_thresh], name + '_vt', kwargs['initializer'], dtype=np.float32)
    create_tensor([-1], name + '_m1', kwargs['initializer'])
    create_tensor([-1], name + '_m1_f', kwargs['initializer'], dtype=dtype)
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([2], name + '_2', kwargs['initializer'])
    create_tensor([3], name + '_3', kwargs['initializer'])
    create_tensor([0, 1, -1], name + '_scores_shape', kwargs['initializer'])
    create_tensor([0, 0, 1, 0], name + '_pad', kwargs['initializer'])
    create_tensor([0, -1], name + '_bat_spat_helper', kwargs['initializer'])
    create_const_scalar_node(name + '_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name + '_1_s', np.int64(1), kwargs)
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Shape', [name + '_shape'], [name + '_dim']), make_node('Sub', [name + '_dim', name + '_2'], [name + '_dim_m2']), make_node('Slice', [name + '_shape', name + '_dim_m2', name + '_dim'], [name + '_shape_last2']), make_node('Concat', [name + '_m1', name + '_shape_last2'], [name + '_shape_3d'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_3d'], [name + '_data_3d']), make_node('Slice', [name + '_data_3d', name + '_cs', name + '_cs_p4', name + '_m1'], [name + '_boxes']), make_node('Slice', [name + '_data_3d', name + '_si', name + '_si_p1', name + '_m1'], [name + '_scores_raw']), make_node('Reshape', [name + '_scores_raw', name + '_scores_shape'], [name + '_scores']), make_node('Shape', [name + '_scores'], [name + '_scores_shape_actual']), make_node('NonMaxSuppression', [name + '_boxes', name + '_scores', name + '_topk', name + '_ot', name + '_vt'], [name + '_nms'], center_point_box=center_point_box), make_node('Slice', [name + '_nms', name + '_0', name + '_3', name + '_m1', name + '_2'], [name + '_nms_sliced']), make_node('GatherND', [name + '_data_3d', name + '_nms_sliced'], [name + '_candidates']), make_node('Pad', [name + '_candidates', name + '_pad', name + '_m1_f'], [name + '_cand_padded']), make_node('Shape', [name + '_nms'], [name + '_nms_shape']), make_node('Slice', [name + '_nms_shape', name + '_0', name + '_1'], [name + '_cand_cnt']), make_node('Squeeze', [name + '_cand_cnt'], [name + '_cc_s'], axes=[0]), make_node('Range', [name + '_0_s', name + '_cc_s', name + '_1_s'], [name + '_cand_indices']), make_node('Slice', [name + '_scores_shape_actual', name + '_0', name + '_3', name + '_m1', name + '_2'], [name + '_shape_bat_spat']), make_node('Slice', [name + '_shape_bat_spat', name + '_1', name + '_2'], [name + '_spat_dim']), make_node('Expand', [name + '_cand_cnt', name + '_shape_bat_spat'], [name + '_base_indices']), make_node('ScatterND', [name + '_base_indices', name + '_nms_sliced', name + '_cand_indices'], [name + '_indices']), make_node('TopK', [name + '_indices', name + '_spat_dim'], [name + '_indices_sorted', name + '__'], largest=0, axis=-1, sorted=1), make_node('Gather', [name + '_cand_padded', name + '_indices_sorted'], [name + '_gather']), make_node('Reshape', [name + '_gather', name + '_shape'], [name + '0'])]
    return nodes