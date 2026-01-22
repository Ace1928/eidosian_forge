from typing import TYPE_CHECKING, Optional, Tuple, Union
def convert_to_pil_image(image: Union['numpy.ndarray', list]) -> 'PIL.Image.Image':
    """
    Convert a numpy array to a PIL image.
    """
    import numpy as np
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError('Pillow is required to serialize a numpy array as an image. Please install it via: `pip install Pillow`') from exc

    def _normalize_to_uint8(x):
        is_int = np.issubdtype(x.dtype, np.integer)
        low = 0
        high = 255 if is_int else 1
        if x.min() < low or x.max() > high:
            if is_int:
                raise ValueError(f'Integer pixel values out of acceptable range [0, 255]. Found minimum value {x.min()} and maximum value {x.max()}. Ensure all pixel values are within the specified range.')
            else:
                raise ValueError(f'Float pixel values out of acceptable range [0.0, 1.0]. Found minimum value {x.min()} and maximum value {x.max()}. Ensure all pixel values are within the specified range.')
        if not is_int:
            x = x * 255
        return x.astype(np.uint8)
    valid_data_types = {'b': 'bool', 'i': 'signed integer', 'u': 'unsigned integer', 'f': 'floating'}
    if image.dtype.kind not in valid_data_types:
        raise TypeError(f"Invalid array data type: '{image.dtype}'. Must be one of {list(valid_data_types.values())}")
    if image.ndim not in [2, 3]:
        raise ValueError(f'`image` must be a 2D or 3D array but got image shape: {image.shape}')
    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
        raise ValueError(f'Invalid channel length: {image.shape[2]}. Must be one of [1, 3, 4]')
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    image = _normalize_to_uint8(image)
    return Image.fromarray(image)