import unittest
import os
from boto.mturk.question import QuestionForm
from .common import MTurkCommon
def testCallCreateHitWithOneQuestion(self):
    create_hit_rs = self.conn.create_hit(question=self.get_question(), **self.get_hit_params())