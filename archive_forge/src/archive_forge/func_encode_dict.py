from typing import Dict, Type, Callable, List
def encode_dict(x, r):
    r.append(b'd')
    ilist = sorted(x.items())
    for k, v in ilist:
        r.extend((int_to_bytes(len(k)), b':', k))
        encode_func[type(v)](v, r)
    r.append(b'e')