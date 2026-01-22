from decimal import Decimal
from kivy.lang import Builder
from kivy.properties import ListProperty
def calculate_cover(self, *args):
    if not self.reference_size:
        return
    size = self.size
    origin_appr = self._aspect_ratio_approximate(self.reference_size)
    crop_appr = self._aspect_ratio_approximate(size)
    if origin_appr == crop_appr:
        crop_size = self.size
        offset = (0, 0)
    elif origin_appr < crop_appr:
        crop_size = self._scale_size(self.reference_size, (size[0], None))
        offset = (0, (crop_size[1] - size[1]) / 2 * -1)
    else:
        crop_size = self._scale_size(self.reference_size, (None, size[1]))
        offset = ((crop_size[0] - size[0]) / 2 * -1, 0)
    self.cover_size = crop_size
    self.cover_pos = offset