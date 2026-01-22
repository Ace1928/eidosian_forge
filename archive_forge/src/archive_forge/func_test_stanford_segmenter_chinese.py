from typing import List, Tuple
import pytest
from nltk.tokenize import (
@check_stanford_segmenter
def test_stanford_segmenter_chinese(self):
    """
        Test the Stanford Word Segmenter for Chinese (default config)
        """
    seg = StanfordSegmenter()
    seg.default_config('zh')
    sent = '这是斯坦福中文分词器测试'
    segmented_sent = seg.segment(sent.split())
    assert segmented_sent.split() == ['这', '是', '斯坦福', '中文', '分词器', '测试']