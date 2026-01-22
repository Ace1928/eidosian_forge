
. rreg stackloss airflow watertemp acidconc

   Huber iteration 1:  maximum difference in weights = .48402478
   Huber iteration 2:  maximum difference in weights = .07083248
   Huber iteration 3:  maximum difference in weights = .03630349
Biweight iteration 4:  maximum difference in weights = .2114744
Biweight iteration 5:  maximum difference in weights = .04709559
Biweight iteration 6:  maximum difference in weights = .01648123
Biweight iteration 7:  maximum difference in weights = .01050023
Biweight iteration 8:  maximum difference in weights = .0027233

Robust regression                                      Number of obs =      21
                                                       F(  3,    17) =   74.15
                                                       Prob > F      =  0.0000

------------------------------------------------------------------------------
   stackloss |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
     airflow |   .8526511   .1223835     6.97   0.000     .5944446    1.110858
   watertemp |   .8733594   .3339811     2.61   0.018     .1687209    1.577998
    acidconc |  -.1224349   .1418364    -0.86   0.400    -.4216836    .1768139
       _cons |   -41.6703   10.79559    -3.86   0.001      -64.447   -18.89361
------------------------------------------------------------------------------

