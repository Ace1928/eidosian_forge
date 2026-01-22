import contextlib
import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import dns.edns
import dns.entropy
import dns.enum
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.OPT
import dns.rdtypes.ANY.TSIG
import dns.renderer
import dns.rrset
import dns.tsig
import dns.ttl
import dns.wire
def section_count(self, section: SectionType) -> int:
    """Returns the number of records in the specified section.

        *section*, an ``int`` section number, a ``str`` section name, or one of
        the section attributes of this message.  This specifies the
        the section of the message to count.  For example::

            my_message.section_count(my_message.answer)
            my_message.section_count(dns.message.ANSWER)
            my_message.section_count("ANSWER")
        """
    if isinstance(section, int):
        section_number = section
        section = self.section_from_number(section_number)
    elif isinstance(section, str):
        section_number = self._section_enum.from_text(section)
        section = self.section_from_number(section_number)
    else:
        section_number = self.section_number(section)
    count = sum((max(1, len(rrs)) for rrs in section))
    if section_number == MessageSection.ADDITIONAL:
        if self.opt is not None:
            count += 1
        if self.tsig is not None:
            count += 1
    return count