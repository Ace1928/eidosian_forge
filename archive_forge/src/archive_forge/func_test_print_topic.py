import numpy as np
def test_print_topic(self):
    topics = self.model.show_topics(formatted=True)
    for topic_no, topic in topics:
        self.assertTrue(isinstance(topic_no, int))
        self.assertTrue(isinstance(topic, str))