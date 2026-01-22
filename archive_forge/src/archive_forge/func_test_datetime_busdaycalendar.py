import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_busdaycalendar(self):
    bdd = np.busdaycalendar(holidays=['NaT', '2011-01-17', '2011-03-06', 'NaT', '2011-12-26', '2011-05-30', '2011-01-17'])
    assert_equal(bdd.holidays, np.array(['2011-01-17', '2011-05-30', '2011-12-26'], dtype='M8'))
    assert_equal(bdd.weekmask, np.array([1, 1, 1, 1, 1, 0, 0], dtype='?'))
    bdd = np.busdaycalendar(weekmask='Sun TueWed  Thu\tFri')
    assert_equal(bdd.weekmask, np.array([0, 1, 1, 1, 1, 0, 1], dtype='?'))
    bdd = np.busdaycalendar(weekmask='0011001')
    assert_equal(bdd.weekmask, np.array([0, 0, 1, 1, 0, 0, 1], dtype='?'))
    bdd = np.busdaycalendar(weekmask='Mon Tue')
    assert_equal(bdd.weekmask, np.array([1, 1, 0, 0, 0, 0, 0], dtype='?'))
    assert_raises(ValueError, np.busdaycalendar, weekmask=[0, 0, 0, 0, 0, 0, 0])
    assert_raises(ValueError, np.busdaycalendar, weekmask='satsun')
    assert_raises(ValueError, np.busdaycalendar, weekmask='')
    assert_raises(ValueError, np.busdaycalendar, weekmask='Mon Tue We')
    assert_raises(ValueError, np.busdaycalendar, weekmask='Max')
    assert_raises(ValueError, np.busdaycalendar, weekmask='Monday Tue')