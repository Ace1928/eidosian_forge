from IPython.testing.ipunittest import ipdoctest, ipdocstring
@ipdocstring
def ipdt_method(self):
    """
        In [20]: print(1)
        1

        In [26]: for i in range(4):
           ....:     print(i)
           ....:     
           ....: 
        0
        1
        2
        3

        In [27]: 3+4
        Out[27]: 7
        """