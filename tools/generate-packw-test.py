#!/usr/bin/env python
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='PackW microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())

def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_x(\d+)_packw_gemm_goi_ukernel_x(\d+)(c(\d+))?(s(\d+))?__(.+)", name)
  assert match is not None
  nr = int(match.group(2))
  if match.group(3):
    kr = int(match.group(4))
  else:
    kr = 1
  if match.group(5):
    sr = int(match.group(6))
  else:
    sr = 1
  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(7))
  return nr, kr, sr, arch, isa


PACKW_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, n_eq_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PackWMicrokernelTester()
    .n(${NR})
    .nr(${NR})
    .kr(${KR})
    .sr(${SR})
    .Test(${", ".join(TEST_ARGS)});
}

$if NR > 1:
  TEST(${TEST_NAME}, n_div_${NR}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t n = ${NR*2}; n < ${NR*10}; n += ${NR}) {
      PackWMicrokernelTester()
        .n(n)
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, n_lt_${NR}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t n = 1; n < ${NR}; n++) {
      PackWMicrokernelTester()
        .n(n)
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, n_gt_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t n = ${NR+1}; n < ${10 if NR == 1 else NR*2}; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, k_eq_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PackWMicrokernelTester()
    .k(${NR})
    .n(${NR})
    .nr(${NR})
    .sr(${SR})
    .Test(${", ".join(TEST_ARGS)});
}

$if NR > 1:
  TEST(${TEST_NAME}, k_div_${NR}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t n = ${NR*2}; n < ${NR*10}; n += ${NR}) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, k_lt_${NR}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t n = 1; n < ${NR}; n++) {
      PackWMicrokernelTester()
        .k(n)
        .n(n)
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, k_gt_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t n = ${NR+1}; n < ${10 if NR == 1 else NR*2}; n++) {
    PackWMicrokernelTester()
      .k(n)
      .n(n)
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, null_bias) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t n = 1; n < ${10 if NR == 1 else NR*2}; n++) {
    PackWMicrokernelTester()
      .n(n)
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .nullbias(true)
      .Test(${", ".join(TEST_ARGS)});
  }
}

"""


def generate_test_cases(ukernel, nr, kr, sr, isa):
  """Generates all tests cases for a PackW micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    nr: NR parameter of the PACKW micro-kernel.
    kr: KR parameter of the PACKW micro-kernel.
    sr: SR parameter of the PACKW micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  return xngen.preprocess(PACKW_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": [ukernel],
      "NR": nr,
      "KR": kr,
      "SR": sr,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/packw.h>
#include "packw-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      nr, kr, sr, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, nr, kr, sr, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    txt_changed = True
    if os.path.exists(options.output):
      with codecs.open(options.output, "r", encoding="utf-8") as output_file:
        txt_changed = output_file.read() != tests

    if txt_changed:
      with codecs.open(options.output, "w", encoding="utf-8") as output_file:
        output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])