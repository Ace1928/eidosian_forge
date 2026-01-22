import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def test_flags(self):
    self.expect_flags('sse sse2 vsx vsx2 neon neon_fp16 vx vxe', x86_gcc='-msse -msse2', x86_icc='-msse -msse2', x86_iccw='/arch:SSE2', x86_msvc='/arch:SSE2' if self.march() == 'x86' else '', ppc64_gcc='-mcpu=power8', ppc64_clang='-mcpu=power8', armhf_gcc='-mfpu=neon-fp16 -mfp16-format=ieee', aarch64='', s390x='-mzvector -march=arch12')
    self.expect_flags('asimd', aarch64='', armhf_gcc='-mfp16-format=ieee -mfpu=neon-fp-armv8 -march=armv8-a\\+simd')
    self.expect_flags('asimdhp', aarch64_gcc='-march=armv8.2-a\\+fp16', armhf_gcc='-mfp16-format=ieee -mfpu=neon-fp-armv8 -march=armv8.2-a\\+fp16')
    self.expect_flags('asimddp', aarch64_gcc='-march=armv8.2-a\\+dotprod')
    self.expect_flags('asimdfhm', aarch64_gcc='-march=armv8.2-a\\+fp16\\+fp16fml')
    self.expect_flags('asimddp asimdhp asimdfhm', aarch64_gcc='-march=armv8.2-a\\+dotprod\\+fp16\\+fp16fml')
    self.expect_flags('vx vxe vxe2', s390x='-mzvector -march=arch13')