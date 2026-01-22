from OpenGL.EGL import *
import itertools
def write_ppm(buf, filename):
    """Write height * width * 3-component buffer as ppm to filename
    
    This lets us write a simple image format without
    using any libraries that can be viewed on most
    linux workstations.
    """
    with open(filename, 'w') as f:
        h, w, c = buf.shape
        f.write('P3\n')
        f.write('# ascii ppm file created by pyopengl\n')
        f.write('%i %i\n' % (w, h))
        f.write('255\n')
        for y in range(h - 1, -1, -1):
            for x in range(w):
                pixel = buf[y, x]
                l = ' %3d %3d %3d' % (pixel[0], pixel[1], pixel[2])
                f.write(l)
            f.write('\n')