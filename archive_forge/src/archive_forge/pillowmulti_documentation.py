import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
Convert image to Paletted PIL image.

        PIL used to not do a very good job at quantization, but I guess
        this has improved a lot (at least in Pillow). I don't think we need
        neuqant (and we can add it later if we really want).
        