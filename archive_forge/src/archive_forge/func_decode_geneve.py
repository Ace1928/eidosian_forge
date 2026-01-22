import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
def decode_geneve(mask, value):
    """Decode geneve options.
    Used for both tnl_push(header(geneve(options()))) action and
    tunnel(geneve()) match.

    It has the following format:

    {class=0xffff,type=0x80,len=4,0xa}

    Args:
        mask (bool): Whether masking is supported.
        value (str): The value to decode.
    """
    if mask:
        decoders = {'class': Mask16, 'type': Mask8, 'len': Mask8}

        def free_decoder(value):
            return ('data', Mask128(value))
    else:
        decoders = {'class': decode_int, 'type': decode_int, 'len': decode_int}

        def free_decoder(value):
            return ('data', decode_int(value))
    result = []
    for opts in re.findall('{.*?}', value):
        result.append(decode_nested_kv(KVDecoders(decoders=decoders, default_free=free_decoder), opts.strip('{}')))
    return result