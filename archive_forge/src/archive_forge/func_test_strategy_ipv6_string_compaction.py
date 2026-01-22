import platform
import pytest
from netaddr import AddrFormatError
from netaddr.strategy import ipv6
@pytest.mark.parametrize(('long_form', 'short_form'), (('FEDC:BA98:7654:3210:FEDC:BA98:7654:3210', 'fedc:ba98:7654:3210:fedc:ba98:7654:3210'), ('1080:0:0:0:8:800:200C:417A', '1080::8:800:200c:417a'), ('FF01:0:0:0:0:0:0:43', 'ff01::43'), ('0:0:0:0:0:0:0:1', '::1'), ('0:0:0:0:0:0:0:0', '::')))
def test_strategy_ipv6_string_compaction(long_form, short_form):
    int_val = ipv6.str_to_int(long_form)
    calc_short_form = ipv6.int_to_str(int_val)
    assert calc_short_form == short_form