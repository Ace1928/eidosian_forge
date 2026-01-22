from sympy.core import backend as sm
import sympy.physics.mechanics as me
def multi_mass_spring_damper(n=1, apply_gravity=False, apply_external_forces=False):
    """Returns a system containing the symbolic equations of motion and
    associated variables for a simple multi-degree of freedom point mass,
    spring, damper system with optional gravitational and external
    specified forces. For example, a two mass system under the influence of
    gravity and external forces looks like:

    ::

        ----------------
         |     |     |   | g
         \\    | |    |   V
      k0 /    --- c0 |
         |     |     | x0, v0
        ---------    V
        |  m0   | -----
        ---------    |
         | |   |     |
         \\ v  | |    |
      k1 / f0 --- c1 |
         |     |     | x1, v1
        ---------    V
        |  m1   | -----
        ---------
           | f1
           V

    Parameters
    ==========

    n : integer
        The number of masses in the serial chain.
    apply_gravity : boolean
        If true, gravity will be applied to each mass.
    apply_external_forces : boolean
        If true, a time varying external force will be applied to each mass.

    Returns
    =======

    kane : sympy.physics.mechanics.kane.KanesMethod
        A KanesMethod object.

    """
    mass = sm.symbols('m:{}'.format(n))
    stiffness = sm.symbols('k:{}'.format(n))
    damping = sm.symbols('c:{}'.format(n))
    acceleration_due_to_gravity = sm.symbols('g')
    coordinates = me.dynamicsymbols('x:{}'.format(n))
    speeds = me.dynamicsymbols('v:{}'.format(n))
    specifieds = me.dynamicsymbols('f:{}'.format(n))
    ceiling = me.ReferenceFrame('N')
    origin = me.Point('origin')
    origin.set_vel(ceiling, 0)
    points = [origin]
    kinematic_equations = []
    particles = []
    forces = []
    for i in range(n):
        center = points[-1].locatenew('center{}'.format(i), coordinates[i] * ceiling.x)
        center.set_vel(ceiling, points[-1].vel(ceiling) + speeds[i] * ceiling.x)
        points.append(center)
        block = me.Particle('block{}'.format(i), center, mass[i])
        kinematic_equations.append(speeds[i] - coordinates[i].diff())
        total_force = -stiffness[i] * coordinates[i] - damping[i] * speeds[i]
        try:
            total_force += stiffness[i + 1] * coordinates[i + 1] + damping[i + 1] * speeds[i + 1]
        except IndexError:
            pass
        if apply_gravity:
            total_force += mass[i] * acceleration_due_to_gravity
        if apply_external_forces:
            total_force += specifieds[i]
        forces.append((center, total_force * ceiling.x))
        particles.append(block)
    kane = me.KanesMethod(ceiling, q_ind=coordinates, u_ind=speeds, kd_eqs=kinematic_equations)
    kane.kanes_equations(particles, forces)
    return kane