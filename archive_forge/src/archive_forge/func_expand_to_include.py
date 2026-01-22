from ctypes import Structure, c_double
from django.contrib.gis.gdal.error import GDALException
def expand_to_include(self, *args):
    """
        Modify the envelope to expand to include the boundaries of
        the passed-in 2-tuple (a point), 4-tuple (an extent) or
        envelope.
        """
    if len(args) == 1:
        if isinstance(args[0], Envelope):
            return self.expand_to_include(args[0].tuple)
        elif hasattr(args[0], 'x') and hasattr(args[0], 'y'):
            return self.expand_to_include(args[0].x, args[0].y, args[0].x, args[0].y)
        elif isinstance(args[0], (tuple, list)):
            if len(args[0]) == 2:
                return self.expand_to_include((args[0][0], args[0][1], args[0][0], args[0][1]))
            elif len(args[0]) == 4:
                minx, miny, maxx, maxy = args[0]
                if minx < self._envelope.MinX:
                    self._envelope.MinX = minx
                if miny < self._envelope.MinY:
                    self._envelope.MinY = miny
                if maxx > self._envelope.MaxX:
                    self._envelope.MaxX = maxx
                if maxy > self._envelope.MaxY:
                    self._envelope.MaxY = maxy
            else:
                raise GDALException('Incorrect number of tuple elements (%d).' % len(args[0]))
        else:
            raise TypeError('Incorrect type of argument: %s' % type(args[0]))
    elif len(args) == 2:
        return self.expand_to_include((args[0], args[1], args[0], args[1]))
    elif len(args) == 4:
        return self.expand_to_include(args)
    else:
        raise GDALException('Incorrect number (%d) of arguments.' % len(args[0]))