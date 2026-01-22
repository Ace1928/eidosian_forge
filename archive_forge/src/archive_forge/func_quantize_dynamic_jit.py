import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def quantize_dynamic_jit(model, qconfig_dict, inplace=False, debug=False):
    """Quantize the input float TorchScript model with
    post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        descriptions in :func:`~torch.ao.quantization.quantize_jit`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchSciprt model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization import quantize_dynamic_jit

    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)

    quantized_model = quantize_dynamic_jit(
        ts_model,
        {'': qconfig},
        calibrate,
        [data_loader_test])
    ```
    """
    torch._C._log_api_usage_once('quantization_api.quantize_jit.quantize_dynamic_jit')
    return _quantize_jit(model, qconfig_dict, inplace=inplace, debug=debug, quant_type=QuantType.DYNAMIC)