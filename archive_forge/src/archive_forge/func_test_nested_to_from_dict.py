from __future__ import annotations
import dataclasses
import datetime
import json
import os
import pathlib
from enum import Enum
import numpy as np
import pandas as pd
import pytest
import torch
from bson.objectid import ObjectId
from monty.json import MontyDecoder, MontyEncoder, MSONable, _load_redirect, jsanitize
from . import __version__ as tests_version
def test_nested_to_from_dict(self):
    GMC = GoodMSONClass
    a_list = [GMC(1, 1.0, 'one'), GMC(2, 2.0, 'two')]
    b_dict = {'first': GMC(3, 3.0, 'three'), 'second': GMC(4, 4.0, 'four')}
    c_list_dict_list = [{'list1': [GMC(5, 5.0, 'five'), GMC(6, 6.0, 'six'), GMC(7, 7.0, 'seven')], 'list2': [GMC(8, 8.0, 'eight')]}, {'list3': [GMC(9, 9.0, 'nine'), GMC(10, 10.0, 'ten'), GMC(11, 11.0, 'eleven'), GMC(12, 12.0, 'twelve')], 'list4': [GMC(13, 13.0, 'thirteen'), GMC(14, 14.0, 'fourteen')], 'list5': [GMC(15, 15.0, 'fifteen')]}]
    obj = GoodNestedMSONClass(a_list=a_list, b_dict=b_dict, c_list_dict_list=c_list_dict_list)
    obj_dict = obj.as_dict()
    obj2 = GoodNestedMSONClass.from_dict(obj_dict)
    assert [obj2.a_list[ii] == aa for ii, aa in enumerate(obj.a_list)]
    assert [obj2.b_dict[kk] == val for kk, val in obj.b_dict.items()]
    assert len(obj.a_list) == len(obj2.a_list)
    assert len(obj.b_dict) == len(obj2.b_dict)
    s = json.dumps(obj_dict)
    obj3 = json.loads(s, cls=MontyDecoder)
    assert [obj2.a_list[ii] == aa for ii, aa in enumerate(obj3.a_list)]
    assert [obj2.b_dict[kk] == val for kk, val in obj3.b_dict.items()]
    assert len(obj3.a_list) == len(obj2.a_list)
    assert len(obj3.b_dict) == len(obj2.b_dict)
    s = json.dumps(obj, cls=MontyEncoder)
    obj4 = json.loads(s, cls=MontyDecoder)
    assert [obj4.a_list[ii] == aa for ii, aa in enumerate(obj.a_list)]
    assert [obj4.b_dict[kk] == val for kk, val in obj.b_dict.items()]
    assert len(obj.a_list) == len(obj4.a_list)
    assert len(obj.b_dict) == len(obj4.b_dict)