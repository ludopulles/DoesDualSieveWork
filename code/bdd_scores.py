#!/usr/bin/python3
"""
Code that executes a dual attack on a small dimension LWE instance and
gathers some statistics about the likeliness the attack will work.
This code computes a lot of scores for BDD-samples at a specific distance rel. to GH.
"""
import argparse
import csv
import datetime
from math import ceil, cos, erfc, exp, log2, pi, sqrt
from multiprocessing import cpu_count, Pool
from time import perf_counter, process_time
from numpy import array, concatenate, linalg, random as np_random
import matplotlib.pyplot as plt

# This code requires an installation of G6K and FPyLLL (pip installable).
# FPLLL, G6K imports
from fpylll import IntegerMatrix, LLL
from g6k import SieverParams
from g6k.siever import Siever


def gh(n):
    """
    Return the expected length of the shortest vector in a random lattice of volume 1.
    """
    return sqrt(n * (pi * n)**(1.0/n) / (2 * pi * exp(1)))


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


def plot_data(sub_plot, outliers):
    """
    Generates a log-plot of the values
    :param outliers: a list of scores.
    """
    xs, ys = [], []

    num_smaller = 0
    for x in range(len(outliers)):
        xs.append(outliers[x])
        num_smaller += 1
        ys.append(log2(num_smaller) - args.bdd)

    sub_plot.plot(xs, ys, color='tab:red', label='real')
    del xs, ys


def write_csv(args, threads, num_dual_vectors, wall_time, cpu_time, real_stats, pred_stats, scores):
    """
    Write the results to a CSV file.
    """
    with open(f"bdd_data_n={args.n}_ghf={args.gh_factor}_bdd={args.bdd}.csv", 'w', newline='') as csvfile:
        fieldnames = ['score', 'cdf_pred', 'cdf_real', 'metakey', 'metaval']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        meta_values = [
            ('n', args.n), ('N', num_dual_vectors), ('log_samples', args.bdd),
            ('gh_factor', args.gh_factor), ('cores', threads), ('walltime', wall_time),
            ('cputime', cpu_time),
            ('real_avg', real_stats["avg"]), ('real_med', real_stats["med"]), ('real_var', real_stats["var"]),
            ('pred_avg', pred_stats["avg"]), ('pred_med', pred_stats["med"]), ('pred_var', pred_stats["var"]),

        ]
        for (key, val) in meta_values:
            writer.writerow({'metakey': key, 'metaval': val})

        num_smaller = 0

        # Record 1024 datapoints maximum
        resolution = 2**max(0, args.bdd - 10)

        for i in range(len(scores)):
            num_smaller += 1
            if i % resolution:
                continue
            x = scores[i]
            pred = round((0.5 * erfc((pred_avg - x) / sqrt(2 * pred_var))), 3)
            real = round((num_smaller) / 2**args.bdd, 3)

            writer.writerow({'score': round(x, 3), 'cdf_pred': pred, 'cdf_real': real})


def plot_independence_prediction(sub_plot, average, variance):
    """
    Plots the prediction of the simulation.
    """
    X, Y = [], []
    for i in range(-500, 501):
        x = i * (12 * sqrt(variance)) / 1000
        y = 0.5 * erfc(-x / sqrt(2 * variance))
        if y > 0:
            X.append(average + x)
            Y.append(log2(y))

    sub_plot.plot(X, Y, linewidth=0.8, color='black', linestyle="dotted", label="pred")


###############################################################################
if __name__ == '__main__':
    # Parse the command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='count', default=0, help='verbosity level')
    parser.add_argument('-j', type=int, default=cpu_count()//2, help='Number of CPU cores to use')
    parser.add_argument('-n', type=int, default=40, help='Dimension of the (q-ary) lattice')
    parser.add_argument('--bdd', type=int, default=10, help='log2 of # BDD targets to take')
    parser.add_argument('--gh_factor', type=float, default=.6, help='width of the bdd error')
    parser.add_argument('--plot', type=bool, default=False, help='Output a plot in addition to csv')
    parser.add_argument('-q', type=int, default=3329, help='Prime to use in the q-ary lattice')
    args = parser.parse_args()

    threads = min(cpu_count(), args.j)
    print(f"Using {threads} CPU cores to grind some data!")

    n = args.n
    gh_factor = args.gh_factor
    q = args.q

    num_samples = 2**args.bdd
    assert num_samples % 2 == 0

    timestamp = perf_counter()

    assert n % 2 == 0

    begin_wall_time = perf_counter()
    begin_cpu_time = process_time()

    B = generate_LWE_lattice(n, n // 2, q)
    # First, divide by sqrt(n) to go to std dev in 1 dimension.
    # Then divide by sqrt(q) to go to volume 1.
    sigma = gh_factor * gh(n) / sqrt(n * q)

    # Construct a sublattice by multiplying the first fft basis vectors by 2.
    timestamp = perf_counter()

    # We get vectors in the primal rather than dual
    # because we can just sample the error in the dual, and thats simpler
    g6k = reduce_and_sieve(B, threads, beta=n, verbose=True)
    g6k.resize_db(ceil(0.9 * (4./3)**(n/2)))

    def change_basis(x):
        """
        Turns a coefficient-vector x into an actual dual vector expressed in canonical basis.
        """
        return array(g6k.M.B.multiply_left(list(x)))

    # Initially, dual vectors are given by coefficients in terms of the (order
    # reversed) dual basis of Bprime.
    # Now U^{-1} Bprime = diag(22...211...1) B
    # Dual vectors are expressed as combinations of (diag(22...211...1)B)^{-T}
    # print(f"Primal sieve took {perf_counter() - timestamp:.2f} seconds.\n")
    db = Pool(threads).map(change_basis, g6k.itervalues())
    N = len(db)

    print(f"Primal sieve took {perf_counter() - timestamp:.2f} seconds.\n")
    timestamp = perf_counter()

    pred_avg, pred_var = 0, 0
    for v in db:
        eps = exp(- 2 * pi**2 * linalg.norm(v)**2 * sigma**2)
        pred_avg += eps
        pred_var += .5 + .5 * eps**4 - eps**2

    def work(seed, num_jobs):
        local_state = np_random.RandomState(seed)
        result = []
        for i in range(num_jobs):
            err = local_state.normal(0, sigma, n)
            s = 0
            for v in db:
                s += cos(2 * pi * err.dot(v))
            result.append(s)
        return result

    jobs = [(i, num_samples // threads + (1 if i < (num_samples % threads) else 0)) for i in range(threads)]

    # Warm up for the benchmark
    for _ in Pool(threads).starmap(work, [(0, 20)] * threads):
        pass
    # Run the benchmark
    start_benchmark = perf_counter()
    for _ in Pool(threads).starmap(work, [(0, 100)] * threads):
        pass
    end_benchmark = perf_counter()
    # Make a prediction on time it will take
    duration_guess = int(ceil(jobs[0][1] / 100 * (end_benchmark - start_benchmark)))
    print(f"Expected time is {duration_guess//3600}h{(duration_guess//60)%60}m{duration_guess%60}s.")
    finish_datetime = datetime.datetime.now() + datetime.timedelta(seconds=duration_guess)
    print(f"Expected completion time at {finish_datetime.strftime('%A %d %B, %H:%M:%S')}")
    # Run the experiment
    scores = concatenate(Pool(threads).starmap(work, jobs))

    print(f"Computing scores took {perf_counter() - timestamp:.2f} seconds.")
    timestamp = perf_counter()

    scores.sort()

    print(f"Sorting   scores took {perf_counter() - timestamp:.2f} seconds.")
    timestamp = perf_counter()

    end_wall_time = perf_counter()
    end_cpu_time = process_time()

    wall_time = end_wall_time - begin_wall_time
    cpu_time = end_cpu_time - begin_cpu_time

    real_avg = sum(scores)/num_samples
    real_var = sum(x**2 for x in scores)/num_samples - real_avg**2
    real_med = 0.5 * (scores[num_samples//2] + scores[num_samples//2 - 1])

    real_stats = {"avg":real_avg, "med": real_med, "var":real_var}
    pred_stats = {"avg":pred_avg, "med": pred_avg, "var":pred_var}

    print("real med, avg, var : ", real_med, real_avg, real_var)
    print("pred med, avg, var : ", pred_avg, pred_avg, pred_var)

    write_csv(args, threads, len(db), wall_time, cpu_time, real_stats, pred_stats, scores)

    if args.plot:
        fig = plt.figure()
        sub_plot = fig.add_subplot(1, 1, 1)

        print(f"Plotting 2^{args.bdd} data points")
        plot_data(sub_plot, scores)
        del scores
        plot_independence_prediction(sub_plot, pred_avg, pred_var)
        sub_plot.legend()
        print("Saving figure...")
        plt.savefig(f"bdd_scores_n={n}_ghf={args.gh_factor}_bdd={args.bdd}.png", dpi=300)
