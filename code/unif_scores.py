#!/usr/bin/python3
"""
Code that executes a dual attack on a small dimension LWE instance and
gathers some statistics about the likeliness the attack will work.
"""
import argparse
import csv
from copy import copy
import ctypes
import datetime
from math import ceil, erfc, exp, log2, sqrt
from multiprocessing import cpu_count, Pool
from time import perf_counter, process_time
from random import randint, random
from numpy import array, concatenate, int32, int64, float64, linalg, zeros
import matplotlib.pyplot as plt

# This code requires an installation of G6K and FPyLLL (pip installable).
# FPLLL, G6K imports
from fpylll import IntegerMatrix, LLL
from g6k import SieverParams
from g6k.siever import Siever


class DiscreteGaussian:
    """
    Sampler for an integer that is distributed by a discrete gaussian of a
    given standard deviation
    """

    def __init__(self, sigma, tau=10):
        """
        Discrete Gaussian Sampler over the integers

        :param sigma: targeted standard deviation (std.dev. of the discrete
        gaussian is close to sigma, but not necessarily equal).
        :param tau: number of standard deviations away at which we will perform
        a tail-cut, i.e. there will not be any value outputted that is larger
        than tau * sigma in absolute value.

        :returns: DiscreteGaussian object
        """
        # The point at which the gaussian's tail is cut:
        self.tail = int(tau * sigma + 2)
        # Create the cumulative density table of length `tail`, where the i'th
        # entry contains `\sum_{|x| <= i} rho_{sigma}(x)`.
        self.cdt = self.tail * [0]

        factor = -0.5 / (sigma * sigma)
        cum_prod = 1.0
        self.cdt[0] = 1.0
        for i in range(1, self.tail):
            # Exploit the symmetry of P(X = x) = P(X = -x) for non-zero x.
            cum_prod += 2 * exp(factor * (i * i))
            self.cdt[i] = cum_prod
        # The total gaussian weight:
        self.renorm = cum_prod

    def __call__(self):
        """
        Takes one sample from the Discrete Gaussian

        :returns: the integer that is the output of the sample, i.e. outputs a
        number `x` with probability exp(-x^2 / 2sigma^2)
        """
        rand = random() * self.renorm
        for i in range(self.tail):
            if rand < self.cdt[i]:
                # The probability to end up here is precisely:
                #     (self.cdt[i] - self.cdt[i-1]) / self.renorm = P(|X| = i)
                # Thus, flip a coin to choose the sign (no effect when i = 0)
                return i * (-1)**randint(0, 1)
        # This should not happen:
        return self.tail * (-1)**randint(0, 1)


def generate_LWE_lattice(m, n, q):
    """
    Generate a basis for a random `q`-ary latice with `n` secret coefficients and `m` samples,
    i.e., it generates a matrix B of the form

        I_n A
        0   I_{m-n},

    where I_k is the k x k identity matrix and A is a n x (m-n) matrix with
    entries uniformly sampled from {0, 1, ..., q-1}.

    :param m: the dimension of the final lattice
    :param n: the number of secret coordinates.
    :param q: the modulus to use with LWE

    :returns: The LLL reduction of the matrix B from above
    """
    return LLL.reduction(IntegerMatrix.random(m, "qary", k=m-n, q=q))


def reduce_and_sieve(B, threads, beta, dual=False, verbose=False):
    """
    Sieve in the primal and find a lot of short vectors using BKZ-beta.

    :param B: the IntegerMatrix object on which to perform the lattice
    reduction. If dual=True, the dual (inverse transpose of B) is used to
    perform lattice reduction on, while the g6k object will still contain the
    primal basis.
    :param beta: integer indicating that the first beta basis vectors of the
    reduced basis should be used to sieve with. If beta <= 0, no sieving will
    be performed.
    :param threads: the number of threads to use in the sieve.
    :param dual: boolean indicating whether or not to reduce & sieve in the
    dual or in the primal sieve.
    :param verbose: boolean indicating whether or not to output the progress of
    the BKZ reduction (and sieving).

    :returns: a g6k object containing the (dual-)reduced basis. If sieving was
    on, the g6k object will contain a list of short vectorsiev
    """
    g6k = Siever(B, SieverParams(threads=threads, dual_mode=dual))
    # Run progressive BKZ up to (incl.) blocksize `beta`:
    if beta > 0:
        if verbose:
            print("\rSieving", end="")
        g6k.initialize_local(0, max(0, beta - 40), beta)
        g6k(alg="gauss")
        while g6k.l > 0:
            # Perform progressive sieving with the `extend_left` operation
            if verbose:
                print("\rSieving [%3d, %3d]" % (g6k.l, g6k.r), end="")
            g6k.extend_left()
            g6k("bgj1" if g6k.r - g6k.l >= 45 else "gauss")
        if g6k.r >= 40:
            with g6k.temp_params(saturation_ratio=.9, db_size_factor=6):
                g6k(alg="hk3")
        if verbose:
            print("\rSieving is done!    ")

    return g6k


def plot_data(sub_plot, log_iters, bucket, outliers):
    xs, ys = [], []

    num_bigger = sum(bucket) + len(outliers)
    for score in range(ceil(threshold)):
        y = log2(num_bigger) - log_iters
        xs.append(score)
        ys.append(y)
        num_bigger -= bucket[score]
    for x in range(len(outliers)):
        xs.append(outliers[x])
        ys.append(log2(num_bigger) - log_iters)
        num_bigger -= 1

    sub_plot.plot(xs, ys, color='tab:red', label='real')
    del xs, ys


def write_csv(args, threads, num_dual_vectors, wall_time, cpu_time, variance, bucket, outliers):
    with open(f"data_n={args.n}_fft={args.fft}_enum={args.enum}.csv", 'w', newline='') as csvfile:
        fieldnames = ['score', 'log2_sf_pred', 'log2_sf_real', 'metakey', 'metaval']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        meta_values = [
            ('n', args.n), ('N', num_dual_vectors), ('fft', args.fft), ('enum', args.enum),
            ('cores', threads), ('walltime', wall_time), ('cputime', cpu_time)
        ]
        for (key, val) in meta_values:
            writer.writerow({'metakey': key, 'metaval': val})

        num_bigger = sum(bucket) + len(outliers)
        for x in range(ceil(threshold)):
            pred = round(log2(0.5 * erfc(x / sqrt(2 * variance))), 3)
            real = round(log2(num_bigger) - (args.enum + args.fft), 3)
            num_bigger -= bucket[x]
            writer.writerow({'score': x, 'log2_sf_pred': pred, 'log2_sf_real': real})

        for i in range(len(outliers)):
            x = outliers[i]
            pred = round(log2(0.5 * erfc(x / sqrt(2 * variance))), 3)
            real = round(log2(num_bigger) - (args.enum + args.fft), 3)
            num_bigger -= 1
            writer.writerow({'score': round(x, 3), 'log2_sf_pred': pred, 'log2_sf_real': real})


def plot_independence_prediction(sub_plot, variance):
    """
    Plots the prediction of the simulation.
    """
    X, Y = [], []
    for i in range(1000):
        x = i * (12 * sqrt(variance)) / 1000
        y = 0.5 * erfc(x / sqrt(2 * variance))
        if y > 0:
            X.append(x)
            Y.append(log2(y))

    sub_plot.plot(X, Y, linewidth=0.8, color='black', linestyle="dotted", label="pred")


###############################################################################
if __name__ == '__main__':
    # Parse the command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='count', default=0, help='verbosity level')
    parser.add_argument('-j', type=int, default=cpu_count()//2, help='Number of CPU cores to use')
    parser.add_argument('-n', type=int, default=40, help='Dimension of the (q-ary) lattice')
    parser.add_argument('--fft', type=int, default=17, help='Dimension of the FFT guessing step')
    parser.add_argument('--enum', type=int, default=10, help='log2 of # random targets to take')
    parser.add_argument('-q', type=int, default=3329, help='Prime to use in the q-ary lattice')
    args = parser.parse_args()

    threads = min(cpu_count(), args.j)
    print(f"Using {threads} CPU cores to grind some data!")

    begin_wall_time = perf_counter()
    begin_cpu_time = process_time()

    B = generate_LWE_lattice(args.n, args.n // 2, args.q)
    # reduce_and_sieve(B, verbose=True)
    B_inv = linalg.inv(array(list(B)))

    # Construct a sublattice by multiplying the first fft basis vectors by 2.
    Bprime = copy(B)
    for i in range(args.fft):
        for j in range(args.n):
            Bprime[i, j] *= 2

    # print(f"Primal sieve took {perf_counter() - timestamp:.2f} seconds.")
    timestamp = perf_counter()

    # HKZ-Reduce the dual of Bprime
    g6k_dual = reduce_and_sieve(Bprime, threads, args.n, dual=True, verbose=(args.v >= 1))
    g6k_dual.resize_db(ceil(0.9 * (4./3)**(args.n/2)))

    def change_basis(x):
        """
        Turns a coefficient-vector x into an actual dual vector expressed in canonical basis.
        """
        return g6k_dual.M.UinvT.multiply_left(list(reversed(x)))

    # Initially, dual vectors are given by coefficients in terms of the (order
    # reversed) dual basis of Bprime.
    # Now U^{-1} Bprime = diag(22...211...1) B
    # Dual vectors are expressed as combinations of (diag(22...211...1)B)^{-T}
    dual_db = Pool(threads).map(change_basis, g6k_dual.itervalues())

    print(f"Dual sieve (n={args.n}) took {perf_counter() - timestamp:.2f} seconds.\n")
    timestamp = perf_counter()

    variance = len(dual_db) * 0.5
    # Only record points away from average (0) by multiple standard deviations.
    threshold = 0.0 * sqrt(variance)
    while (2**(args.fft + args.enum) * erfc(threshold / sqrt(variance * 2.0)) >= 1e3):
        threshold += 0.1 * sqrt(variance)
    BUCKET_SIZE = int(20 * sqrt(variance))
    print(f"bucket size = {BUCKET_SIZE}, threshold = {threshold/sqrt(variance):.2f} \\sigma = {threshold:.2f}")

    def get_random_target():
        """
        Return a uniformly random target that is expressed in basis B from (Z/qZ)^n.
        """
        return array([randint(0, args.q - 1) for _ in range(args.n)], dtype=float64).dot(B_inv)

    def work(num_jobs):
        buckets = zeros(BUCKET_SIZE, dtype=int64)
        buckets_p = buckets.ctypes.data_as(c_int64_p)

        fft_result = zeros(1 << args.fft, dtype=float64)
        fft_result_p = fft_result.ctypes.data_as(c_float_p)

        outliers = []
        for _ in range(num_jobs):
            target = get_random_target()
            target_p = target.ctypes.data_as(c_float_p)

            # Call the C function
            c_binding.argtypes = [ctypes.c_int32, ctypes.c_double,
                                  c_float_p, c_float_p, c_int64_p]
            result = c_binding.FFT_scores(ctypes.c_int32(args.fft), ctypes.c_double(threshold),
                                          target_p, fft_result_p, buckets_p)
            outliers.append(copy(fft_result[:result]))
        del fft_result

        outliers = concatenate(outliers)
        outliers.sort()
        return buckets, outliers

    global_bucket = zeros(BUCKET_SIZE, dtype=int64)
    global_outliers = []

    # Initialize the binding to dual_utils.c
    c_binding = ctypes.CDLL("./dual_utils.so")
    c_int32_p = ctypes.POINTER(ctypes.c_int32)
    c_int64_p = ctypes.POINTER(ctypes.c_int64)
    c_float_p = ctypes.POINTER(ctypes.c_double)

    enc_dual_db = zeros(len(dual_db) * args.n, dtype=int32)
    for (i, dual_vector) in enumerate(dual_db):
        for j in range(args.n):
            enc_dual_db[i * args.n + j] = dual_vector[j]
    c_binding.init_dual_database(len(dual_db), args.n, enc_dual_db.ctypes.data_as(c_int32_p))

    jobs = 2**args.enum
    jobs = [jobs // threads + (1 if i < (jobs % threads) else 0) for i in range(threads)]

    # Warm up for the benchmark
    for (buckets, outliers) in Pool(threads).imap_unordered(work, [20] * threads):
        pass
    # Run the benchmark
    start_benchmark = perf_counter()
    for (buckets, outliers) in Pool(threads).imap_unordered(work, [200] * threads):
        pass
    end_benchmark = perf_counter()
    # Make a prediction on time it will take
    duration_guess = int(ceil(jobs[0] / 200 * (end_benchmark - start_benchmark)))
    print(f"Expected time is {duration_guess//3600}h{(duration_guess//60)%60}m{duration_guess%60}s.")
    finish_datetime = datetime.datetime.now() + datetime.timedelta(seconds=duration_guess)
    print(f"Expected completion time at {finish_datetime.strftime('%A %d %B, %H:%M:%S')}")
    # Run the experiment
    for (buckets, outliers) in Pool(threads).imap_unordered(work, jobs):
        for i in range(BUCKET_SIZE):
            global_bucket[i] += buckets[i]
        global_outliers.append(outliers)
    global_outliers = concatenate(global_outliers)
    global_outliers.sort()
    del outliers, buckets

    c_binding.clean_up()
    del c_binding

    duration = perf_counter() - timestamp
    print(f"Computing scores took {duration:.0f} seconds.")

    end_wall_time = perf_counter()
    end_cpu_time = process_time()

    wall_time = end_wall_time - begin_wall_time
    cpu_time = end_cpu_time - begin_cpu_time

    fig = plt.figure()
    sub_plot = fig.add_subplot(1, 1, 1)
    print(f"Plotting {ceil(threshold)} + {len(global_outliers)} data points")
    plot_data(sub_plot, args.fft + args.enum, global_bucket, global_outliers)
    write_csv(args, threads, len(dual_db), wall_time, cpu_time, variance,
              global_bucket, global_outliers)

    plot_independence_prediction(sub_plot, variance)
    sub_plot.legend()
    print("Saving figure...")
    plt.savefig(f"scores_n={args.n}_fft={args.fft}_enum={args.enum}.png", dpi=300)
