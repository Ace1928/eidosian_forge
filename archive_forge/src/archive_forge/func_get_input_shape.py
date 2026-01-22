from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def get_input_shape(sym, proto_obj):
    """Helper function to obtain the shape of an array"""
    arg_params = proto_obj.arg_dict
    aux_params = proto_obj.aux_dict
    model_input_shape = [data[1] for data in proto_obj.model_metadata.get('input_tensor_data')]
    data_names = [data[0] for data in proto_obj.model_metadata.get('input_tensor_data')]
    inputs = []
    for in_shape in model_input_shape:
        inputs.append(nd.ones(shape=in_shape))
    data_shapes = []
    for idx, input_name in enumerate(data_names):
        data_shapes.append((input_name, inputs[idx].shape))
    ctx = context.cpu()
    mod = module.Module(symbol=sym, data_names=data_names, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    data_forward = []
    for idx, input_name in enumerate(data_names):
        val = inputs[idx]
        data_forward.append(val)
    mod.forward(io.DataBatch(data_forward))
    result = mod.get_outputs()[0].asnumpy()
    return result.shape