import gast as ast
import numpy as np
import numbers
def size_container_folding(value):
    """
    Convert value to ast expression if size is not too big.

    Converter for sized container.
    """

    def size(x):
        return len(getattr(x, 'flatten', lambda: x)())
    if size(value) < MAX_LEN:
        if isinstance(value, list):
            return ast.List([to_ast(elt) for elt in value], ast.Load())
        elif isinstance(value, tuple):
            return ast.Tuple([to_ast(elt) for elt in value], ast.Load())
        elif isinstance(value, set):
            if value:
                return ast.Set([to_ast(elt) for elt in value])
            else:
                return ast.Call(func=ast.Attribute(ast.Name(mangle('builtins'), ast.Load(), None, None), 'set', ast.Load()), args=[], keywords=[])
        elif isinstance(value, dict):
            keys = [to_ast(elt) for elt in value.keys()]
            values = [to_ast(elt) for elt in value.values()]
            return ast.Dict(keys, values)
        elif isinstance(value, np.ndarray):
            if np.ndim(value) == 0 or len(value) == 0:
                shp = to_ast(value.shape)
                shp = ast.Call(ast.Attribute(ast.Attribute(ast.Name(mangle('builtins'), ast.Load(), None, None), 'pythran', ast.Load()), 'make_shape', ast.Load()), args=getattr(shp, 'elts', [shp]), keywords=[])
                return ast.Call(func=ast.Attribute(ast.Name(mangle('numpy'), ast.Load(), None, None), 'empty', ast.Load()), args=[shp, dtype_to_ast(value.dtype.name)], keywords=[])
            else:
                return ast.Call(func=ast.Attribute(ast.Name(mangle('numpy'), ast.Load(), None, None), 'array', ast.Load()), args=[to_ast(totuple(value.tolist())), dtype_to_ast(value.dtype.name)], keywords=[])
        else:
            raise ConversionError()
    else:
        raise ToNotEval()