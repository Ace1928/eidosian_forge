from __future__ import print_function
from Universe import Icecream, Truck
def getFlavor(self):
    base_flavor = super(VanillaChocolateCherryIcecream, self).getFlavor()
    return base_flavor + ' and a cherry'