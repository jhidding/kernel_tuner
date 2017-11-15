"""
The Noodles runner allows tuning in parallel using multiple processes/threads.
"""

import subprocess
import random
from collections import OrderedDict
from itertools import (chain, repeat)

import numpy
import resource
import logging

import noodles
from noodles import schedule_hint, gather, lift
from noodles.run.runners import run_parallel_with_display, run_parallel
from noodles.display import NCDisplay

from noodles.run.queue import (Queue)
from noodles.run.thread_pool import (thread_pool)
from noodles.run.worker import (worker)
from noodles.run.hybrid import run_hybrid
from noodles.draw_workflow import draw_workflow

from ..core import DeviceInterface
from .. import util


def _error_filter(errortype, value=None, tb=None):
    if errortype is subprocess.CalledProcessError:
        return value.stderr
    elif "cuCtxSynchronize" in str(value):
        return value
    return None


def _chunk_list(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def device_worker():
    return Queue() >> worker


def host_worker(n_threads):
    return Queue() >> thread_pool(*repeat(worker, n_threads))


def queue_selector(job):
    if job.hints and 'queue' in job.hints:
        return job.hints['queue']
    else:
        return 'host'


def run_workflow(wf, n_threads):
    """Custom runner for Kernel-tuner workflows."""
    return run_hybrid(
        wf, queue_selector,
        {
            'host':   host_worker(n_threads),
            'device': device_worker()
        })


class Compiler:
    def __init__(self, device):
        self.device = device

    def __deepcopy__(self, memo):
        return self

    def __call__(self, gpu_args_ref, params, kernel_options, tuning_options):
        """ Compile and benchmark a kernel instance based on kernel strings and parameters """
        gpu_args = gpu_args_ref.unref()
        instance_string = util.get_instance_string(params)

        logging.debug('compile_and_benchmark ' + instance_string)
        mem_usage = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1)
        logging.debug('Memory usage : %2.2f MB', mem_usage)

        verbose = tuning_options.verbose

        instance = self.device.create_kernel_instance(kernel_options, params, verbose)
        if instance is None:
            return self, noodles.Fail(
                    self, exception=RuntimeError("Could not create kernel instance."))

        try:
            #compile the kernel
            func = self.device.compile_kernel(instance, verbose)
            if func is None:
                return self, noodles.Fail(
                        self, exception=RuntimeError("Could not compile kernel."))

            #add constant memory arguments to compiled module
            if kernel_options.cmem_args is not None:
                self.device.copy_constant_memory_args(kernel_options.cmem_args)

        except Exception as e:
            #dump kernel_string to temp file
            temp_filename = util.get_temp_filename(suffix=".c")
            util.write_file(temp_filename, instance.kernel_string)
            print("Error while compiling, see source files: " + temp_filename + " ".join(instance.temp_files.values()))
            raise e

        return self, Kernel(self.device, instance, func)


@schedule_hint(queue='device')
@noodles.maybe
def device_call(f, *args):
    return f(*args)


@schedule_hint(queue='host')
@noodles.maybe
def host_call(f, *args):
    return f(*args)


class Kernel:
    def __init__(self, device, instance, func):
        self.device = device
        self.instance = instance
        self.func = func

    def __deepcopy__(self, memo):
        return self

    def __call__(self, gpu_args_ref, tuning_options):
        gpu_args = gpu_args_ref.unref()
        verbose = tuning_options.verbose

        try:
            #test kernel for correctness and benchmark
            if tuning_options.answer is not None:
                self.device.check_kernel_correctness(
                    self.func, gpu_args, self.instance, tuning_options.answer,
                    tuning_options.atol, tuning_options.verify, verbose)

            #benchmark
            time = self.device.benchmark(self.func, gpu_args, self.instance, tuning_options.times, verbose)

        except Exception as e:
            #dump kernel_string to temp file
            temp_filename = util.get_temp_filename(suffix=".c")
            util.write_file(temp_filename, self.instance.kernel_string)
            print("Error while benchmarking, see source files: "
                + temp_filename + " ".join(self.instance.temp_files.values()))
            raise e

        #clean up any temporary files, if no error occured
        for v in self.instance.temp_files.values():
            util.delete_temp_file(v)

        return time


def chunk_prepare(chunk, kernel_options, device_options, tuning_options):
    """Benchmark a single kernel instance in the parameter space, using
    a list of settings, chunked together."""

    # detect language and create high-level device interface
    device = DeviceInterface(
            kernel_options.kernel_string,
            iterations=tuning_options.iterations,
            **device_options)
    compile_kernel = Compiler(device)

    # move data to the GPU
    gpu_args = device.ready_argument_list(kernel_options.arguments)

    def prepare_single(element):
        nonlocal compile_kernel
        params = noodles.simple_lift(OrderedDict(zip(tuning_options.tune_params.keys(), element)))
        compile_kernel, benchmark_kernel = noodles.unpack(
            host_call(compile_kernel, noodles.ref(gpu_args), params, kernel_options, tuning_options), 2)
        time = device_call(benchmark_kernel, noodles.ref(gpu_args), tuning_options)
        params['time'] = time
        return params

    return (prepare_single(element) for element in chunk)


def parameter_sweep(
        parameter_space, kernel_options, device_options,
        tuning_options, max_threads):
    """Build a Noodles workflow by sweeping the parameter space"""
    # randomize parameter space to do pseudo load balancing
    parameter_space = list(parameter_space)
    random.shuffle(parameter_space)

    # split parameter space into chunks
    work_per_thread = len(parameter_space) // max_threads
    work = chain.from_iterable(
        chunk_prepare(chunk, kernel_options, device_options, tuning_options)
        for chunk in _chunk_list(parameter_space, work_per_thread))

    return noodles.gather_all(work)


class NoodlesRunner:
    def __init__(self, device_options, max_threads=1):
        self.device_options = device_options
        self.device_options["quiet"] = True
        self.max_threads = max_threads
        self.dev = None

    def run(self, parameter_space, kernel_options, tuning_options):
        """ Tune all instances in parameter_space using a multiple threads

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param tuning_options: A dictionary with all options regarding the tuning
            process.
        :type tuning_options: kernel_tuner.interface.Options

        :returns: A list of dictionaries for executed kernel configurations and their
            execution times. And a dictionary that contains a information
            about the hardware/software environment on which the tuning took place.
        :rtype: list(dict()), dict()
        """
        workflow = parameter_sweep(
            parameter_space, kernel_options, self.device_options,
            tuning_options, self.max_threads * 2)

        draw_workflow("kernel-tuner-wf.pdf", workflow._workflow)

        if tuning_options.verbose:
            # FIXME : sensible display routine
            result = run_workflow(workflow, self.max_threads)
        else:
            result = run_workflow(workflow, self.max_threads)

        for r in result:
            print(r['time'])

        # Filter out None times
        return [r for r in result if r['time']], {}
