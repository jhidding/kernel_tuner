import sys
import numpy
import kernel_tuner


def assert_noodles_works():
    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 5):
        raise RuntimeError("Noodles runner test requires Python 3.5 or newer")

    import importlib.util
    noodles_installed = importlib.util.find_spec("noodles") is not None

    if not noodles_installed:
        raise RuntimeError("Noodles runner test requires Noodles")


def test_noodles_opencl():
    kernel_string = """
    __kernel void vector_add(global float *c, global float *a, global float *b, int n) {
        int i = get_group_id(0) * block_size_x + get_local_id(0);
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 1000000
    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]
    tune_params = {"block_size_x": [16+16*i for i in range(32)]}

    result, _ = kernel_tuner.tune_kernel(
        "vector_add", kernel_string, size, args, tune_params, quiet=True)
    #    use_noodles=True, num_threads=4, quiet=True)

    print("# {par1:20} | {time:20} ".format(par1='block-size-x', time='time'))
    print("#{bar}|{bar}".format(bar='-'*22))
    for r in sorted(result, key=lambda r: r['block_size_x']):
        print("  {par1:20}   {time:20.6} ".format(par1=r['block_size_x'], time=r['time']))


if __name__ == "__main__":
    assert_noodles_works()
    test_noodles_opencl()
