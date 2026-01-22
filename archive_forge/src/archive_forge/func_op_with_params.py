import sys
import cv2 as cv
def op_with_params(cls):
    if not in_types:
        raise Exception('{} operation should have at least one input!'.format(cls.__name__))
    if not out_types:
        raise Exception('{} operation should have at least one output!'.format(cls.__name__))
    for i, t in enumerate(out_types):
        if t not in [cv.GMat, cv.GScalar, *garray_types, *gopaque_types]:
            raise Exception('{} unsupported output type: {} in position: {}'.format(cls.__name__, t.__name__, i))

    def on(*args):
        if len(in_types) != len(args):
            raise Exception('Invalid number of input elements!\nExpected: {}, Actual: {}'.format(len(in_types), len(args)))
        for i, (t, a) in enumerate(zip(in_types, args)):
            if t in garray_types:
                if not isinstance(a, cv.GArrayT):
                    raise Exception('{} invalid type for argument {}.\nExpected: {}, Actual: {}'.format(cls.__name__, i, cv.GArrayT.__name__, type(a).__name__))
                elif a.type() != garray_types[t]:
                    raise Exception('{} invalid GArrayT type for argument {}.\nExpected: {}, Actual: {}'.format(cls.__name__, i, type2str[garray_types[t]], type2str[a.type()]))
            elif t in gopaque_types:
                if not isinstance(a, cv.GOpaqueT):
                    raise Exception('{} invalid type for argument {}.\nExpected: {}, Actual: {}'.format(cls.__name__, i, cv.GOpaqueT.__name__, type(a).__name__))
                elif a.type() != gopaque_types[t]:
                    raise Exception('{} invalid GOpaque type for argument {}.\nExpected: {}, Actual: {}'.format(cls.__name__, i, type2str[gopaque_types[t]], type2str[a.type()]))
            elif t != type(a):
                raise Exception('{} invalid input type for argument {}.\nExpected: {}, Actual: {}'.format(cls.__name__, i, t.__name__, type(a).__name__))
        op = cv.gapi.__op(op_id, cls.outMeta, *args)
        out_protos = []
        for i, out_type in enumerate(out_types):
            if out_type == cv.GMat:
                out_protos.append(op.getGMat())
            elif out_type == cv.GScalar:
                out_protos.append(op.getGScalar())
            elif out_type in gopaque_types:
                out_protos.append(op.getGOpaque(gopaque_types[out_type]))
            elif out_type in garray_types:
                out_protos.append(op.getGArray(garray_types[out_type]))
            else:
                raise Exception("In {}: G-API operation can't produce the output with type: {} in position: {}".format(cls.__name__, out_type.__name__, i))
        return tuple(out_protos) if len(out_protos) != 1 else out_protos[0]
    cls.id = op_id
    cls.on = staticmethod(on)
    return cls