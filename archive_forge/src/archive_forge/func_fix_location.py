from fontTools.varLib.models import VariationModel, normalizeValue, piecewiseLinearMap
def fix_location(self, location):
    location = dict(location)
    for tag, axis in self.axes_dict.items():
        if tag not in location:
            location[tag] = axis.defaultValue
    return location