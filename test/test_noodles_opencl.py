from .context import skip_if_no_opencl
import sys
import numpy

import kernel_tuner


def test_noodles_opencl():
    skip_if_no_opencl()

    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 5):
        raise SkipTest("Noodles runner test requires Python 3.5 or newer")

    import importlib.util
    noodles_installed = importlib.util.find_spec("noodles") is not None

    if not noodles_installed:
        raise SkipTest("Noodles runner test requires Noodles")

    kernel_string = """
    __kernel void vector_add(global float *c, global float *a, global float *b, int n) {
        int i = get_group_id(0) * block_size_x + get_local_id(0);
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 10000
    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]
    tune_params = {"block_size_x": [16+16*i for i in range(32)]}

    result, _ = kernel_tuner.tune_kernel(
        "vector_add", kernel_string, size, args, tune_params,
        use_noodles=True, num_threads=4)

    assert len(result) == len(tune_params["block_size_x"])
