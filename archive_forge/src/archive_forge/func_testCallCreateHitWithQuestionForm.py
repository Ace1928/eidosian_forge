import unittest
import os
from boto.mturk.question import QuestionForm
from .common import MTurkCommon
def testCallCreateHitWithQuestionForm(self):
    create_hit_rs = self.conn.create_hit(questions=QuestionForm([self.get_question()]), **self.get_hit_params())