from colorsys import hls_to_rgb
def index_to_hue(self, index):
    num, den = (0, 1)
    while index:
        num = num << 1
        den = den << 1
        if index & 1:
            num += 1
        index = index >> 1
    return float(num) / float(den)