import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
def converToPIL(self, im, quantizer, palette_size=256):
    """Convert image to Paletted PIL image.

        PIL used to not do a very good job at quantization, but I guess
        this has improved a lot (at least in Pillow). I don't think we need
        neuqant (and we can add it later if we really want).
        """
    im_pil = ndarray_to_pil(im, 'gif')
    if quantizer in ('nq', 'neuquant'):
        nq_samplefac = 10
        im_pil = im_pil.convert('RGBA')
        nqInstance = NeuQuant(im_pil, nq_samplefac)
        im_pil = nqInstance.quantize(im_pil, colors=palette_size)
    elif quantizer in (0, 1, 2):
        if quantizer == 2:
            im_pil = im_pil.convert('RGBA')
        else:
            im_pil = im_pil.convert('RGB')
        im_pil = im_pil.quantize(colors=palette_size, method=quantizer)
    else:
        raise ValueError('Invalid value for quantizer: %r' % quantizer)
    return im_pil