import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
Prepares the input float TorchScript model with
    *on-device* post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        `method_name`: Name of the method within the model, to be prepared for quantization
        descriptions in :func:`~torch.ao.quantization.quantize_jit`
        `inplace`: carry out model transformations in-place, the original module is
        mutated

    Return:
        TorchScript model that is ready for on device quantization.
        This means that the returned
        model has:
        - Method is inlined.
        - Model has observer modules inserted in the model.
        - Model has packed params inserted in the model. However they are empty as in they dont
          contain valid quantized weights.
        - observe_<method_name> is added that observe the values to be quantized.
        - reset_observers_<method_name> to reset observers.
        - quantize_<method_name> is added to the model.
          - This method extract scale, zero points.
          - Quantizes observed weights.
          - Creates packed params from it and update the attribute of the model with the new values
            for the packed params.
          - Reset the original fp32 weights with empty tensor using SetAttr.
        - quantized_<method_name> is added to the model.
          - This method uses quantized weights and quantized linear ops instead of fp32 op.
          - This method should be used for inference post PTQ.
        - Note that all method's signatures should be the same as method_name.

        Later on device:
        - Run reset_observers_<method_name>
        - Run observe_<method_name>
        - Run quantize_<method_name>
        - Now model can be saved and loaded later.
        - Run model with quantized_<method_name>

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization.quantize_jit import _quantize_ondevice_dynamic_jit

    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig('fbgemm')
    quant_ready_model = _quantize_ondevice_dynamic_jit(
        ts_model,
        {'': qconfig},
        'forward',
        True)
    ```
    