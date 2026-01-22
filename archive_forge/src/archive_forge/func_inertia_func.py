import collections
import warnings
from sympy.external import import_module
def inertia_func(self, v1, v2, l, frame):
    if self.type2[v1] == 'particle':
        l.append('_me.inertia_of_point_mass(' + self.bodies[v1] + '.mass, ' + self.bodies[v1] + '.point.pos_from(' + self.symbol_table2[v2] + '), ' + frame + ')')
    elif self.type2[v1] == 'bodies':
        if self.inertia_point[v1] == v1 + 'o':
            if v2 == self.inertia_point[v1]:
                l.append(self.symbol_table2[v1] + '.inertia[0]')
            else:
                l.append(self.bodies[v1] + '.inertia[0]' + ' + ' + '_me.inertia_of_point_mass(' + self.bodies[v1] + '.mass, ' + self.bodies[v1] + '.masscenter' + '.pos_from(' + self.symbol_table2[v2] + '), ' + frame + ')')
        elif v2 == self.inertia_point[v1]:
            l.append(self.symbol_table2[v1] + '.inertia[0]')
        elif v2 == v1 + 'o':
            l.append(self.bodies[v1] + '.inertia[0]' + ' - ' + '_me.inertia_of_point_mass(' + self.bodies[v1] + '.mass, ' + self.bodies[v1] + '.masscenter' + '.pos_from(' + self.symbol_table2[self.inertia_point[v1]] + '), ' + frame + ')')
        else:
            l.append(self.bodies[v1] + '.inertia[0]' + ' - ' + '_me.inertia_of_point_mass(' + self.bodies[v1] + '.mass, ' + self.bodies[v1] + '.masscenter' + '.pos_from(' + self.symbol_table2[self.inertia_point[v1]] + '), ' + frame + ')' + ' + ' + '_me.inertia_of_point_mass(' + self.bodies[v1] + '.mass, ' + self.bodies[v1] + '.masscenter' + '.pos_from(' + self.symbol_table2[v2] + '), ' + frame + ')')