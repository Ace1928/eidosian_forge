from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_unjoin_explicit_key_3():
    table4 = (('Tournament', 'Year', 'Winner', 'Date of Birth'), ('Indiana Invitational', 1998, 'Al Fredrickson', '21 July 1975'), ('Cleveland Open', 1999, 'Bob Albertson', '28 September 1968'), ('Des Moines Masters', 1999, 'Al Fredrickson', '21 July 1975'), ('Indiana Invitational', 1999, 'Chip Masterson', '14 March 1977'))
    expect_left = (('Tournament', 'Year', 'Winner'), ('Cleveland Open', 1999, 'Bob Albertson'), ('Des Moines Masters', 1999, 'Al Fredrickson'), ('Indiana Invitational', 1998, 'Al Fredrickson'), ('Indiana Invitational', 1999, 'Chip Masterson'))
    expect_right = (('Winner', 'Date of Birth'), ('Al Fredrickson', '21 July 1975'), ('Bob Albertson', '28 September 1968'), ('Chip Masterson', '14 March 1977'))
    left, right = unjoin(table4, 'Date of Birth', key='Winner')
    ieq(expect_left, left)
    ieq(expect_left, left)
    ieq(expect_right, right)
    ieq(expect_right, right)