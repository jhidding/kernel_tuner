#!/usr/bin/env python
import numpy
import kernel_tuner
from collections import OrderedDict
from contextlib import redirect_stdout
import sys

def assert_noodles_works():
    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 5):
        raise RuntimeError("Noodles runner test requires Python 3.5 or newer")

    import importlib.util
    noodles_installed = importlib.util.find_spec("noodles") is not None

    if not noodles_installed:
        raise RuntimeError("Noodles runner test requires Noodles")


if __name__ == "__main__":
    with open('convolution.cl', 'r') as f:
        kernel_string = f.read()

    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    input_size = (problem_size[0]+16) * (problem_size[0]+16)

    output = numpy.zeros(size).astype(numpy.float32)
    input = numpy.random.randn(input_size).astype(numpy.float32)
    filter = numpy.random.randn(17*17).astype(numpy.float32)

    args = [output, input, filter]
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16*i for i in range(1,4)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    with redirect_stdout(sys.stderr):
        result, _ = kernel_tuner.tune_kernel("convolution_kernel", kernel_string,
            problem_size, args, tune_params,
            grid_div_y=grid_div_y, grid_div_x=grid_div_x, quiet=True,
            use_noodles=True, num_threads=2)

    header = "# " + " | ".join("{:15}".format(k) for k in result[0].keys()) + "\n" \
        + "#-" + "-|-".join('-'*15 for _ in result[0].keys())
    print(header)
    line = "  " + "   ".join("{{{}:15}}".format(k) for k in result[0].keys())
    for r in sorted(result, key=lambda r: r['block_size_x']):
        print(line.format(**r))
