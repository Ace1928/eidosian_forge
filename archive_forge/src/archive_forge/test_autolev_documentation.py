import os
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.parsing.autolev import parse_autolev
Autolev example calculates the position, velocity, and acceleration of a
    point and expresses in a single reference frame::

          (1) FRAMES C,D,F
          (2) VARIABLES FD'',DC''
          (3) CONSTANTS R,L
          (4) POINTS O,E
          (5) SIMPROT(F,D,1,FD)
       -> (6) F_D = [1, 0, 0; 0, COS(FD), -SIN(FD); 0, SIN(FD), COS(FD)]
          (7) SIMPROT(D,C,2,DC)
       -> (8) D_C = [COS(DC), 0, SIN(DC); 0, 1, 0; -SIN(DC), 0, COS(DC)]
          (9) W_C_F> = EXPRESS(W_C_F>, F)
       -> (10) W_C_F> = FD'*F1> + COS(FD)*DC'*F2> + SIN(FD)*DC'*F3>
          (11) P_O_E>=R*D2>-L*C1>
          (12) P_O_E>=EXPRESS(P_O_E>, D)
       -> (13) P_O_E> = -L*COS(DC)*D1> + R*D2> + L*SIN(DC)*D3>
          (14) V_E_F>=EXPRESS(DT(P_O_E>,F),D)
       -> (15) V_E_F> = L*SIN(DC)*DC'*D1> - L*SIN(DC)*FD'*D2> + (R*FD'+L*COS(DC)*DC')*D3>
          (16) A_E_F>=EXPRESS(DT(V_E_F>,F),D)
       -> (17) A_E_F> = L*(COS(DC)*DC'^2+SIN(DC)*DC'')*D1> + (-R*FD'^2-2*L*COS(DC)*DC'*FD'-L*SIN(DC)*FD'')*D2> + (R*FD''+L*COS(DC)*DC''-L*SIN(DC)*DC'^2-L*SIN(DC)*FD'^2)*D3>

    