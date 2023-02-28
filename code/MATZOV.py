#!/usr/bin/python3
"""
Code that executes a dual attack on a small dimension LWE instance and
gathers some statistics about the likeliness the attack will work.
"""
import argparse
from functools import partial
from math import ceil, comb, cos, erf, exp, pi, prod, sin, sqrt
from multiprocessing import cpu_count, Pool
from random import random, randint
from statistics import median
from sys import stdout
import matplotlib.pyplot as plt

import numpy as np

# This code requires an installation of G6K and FPyLLL (pip installable).
# FPLLL, G6K imports
from fpylll import IntegerMatrix, BKZ
from fpylll.algorithms.bkz2 import BKZReduction
from g6k import SieverParams
from g6k.siever import Siever


def GH(d):
    """
    Returns the expected length of the shortest vector in a lattice of dimension d and volume 1.
    """
    return sqrt(d * (pi * d)**(1.0/d) / (2 * pi * exp(1)))


class CentredBinomial:
    """
    Sampler for an integer that is distributed by a binomial with
        (n, p) = (2*eta, 0.5),
    but centred to have an outcome in [-eta, eta].
    """
    def __init__(self, eta=3):
        self.eta = eta

    def support(self):
        """
        Give the interval [l, r] on which the PDF is nonzero.
        """
        return range(-self.eta, self.eta + 1)

    def PDF(self, outcome):
        return 0.25**(self.eta) * comb(2*self.eta, outcome + self.eta)

    def __call__(self):
        return sum(randint(0, 1) for i in range(2*self.eta)) - self.eta


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

    def support(self):
        """
        Give the range [l, r] on which the PDF is nonzero.
        """
        return range(1 - self.tail, self.tail)

    def PDF(self, outcome):
        """
        Give the probability on a certain outcome
        """
        if outcome == 0:
            return 1.0 / self.renorm
        return (self.cdt[abs(outcome)] - self.cdt[abs(outcome) - 1]) / self.renorm

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


def variance(distribution):
    """
    Return the variance of the supplied distribution.
    """
    E_X, E_X2 = 0, 0
    for x in distribution.support():
        p = distribution.PDF(x)
        E_X += x*p
        E_X2 += x*x*p
    return E_X2 - E_X**2


def generate_LWE_lattice(m, n, q):
    """
    Generate a basis for a random `q`-ary latice with `n` secret coefficients and `m` samples,
    i.e., it generates a matrix B of the form

        I_n A
        0   q I_{m-n},

    where I_k is the k x k identity matrix and A is a n x (m-n) matrix with
    entries uniformly sampled from {0, 1, ..., q-1}.

    :param m: the dimension of the final lattice
    :param n: the number of secret coordinates.
    :param q: the modulus to use with LWE

    :returns: The matrix A and B from above
    """
    B = IntegerMatrix.random(m, "qary", k=m-n, q=q)
    A = B.submatrix(0, n, n, m)
    return A, B


def progressive_BKZ(B: IntegerMatrix, beta, params: SieverParams, verbose=False):
    """
    Run progressive BKZ up to blocksize beta

    :param B: the IntegerMatrix object on which to perform the lattice
    reduction.  Note that this function changes B.
    :param beta: blocksize up to which inclusive to perform progressive BKZ reduction.
    :param params: The SieverParams with which to instantiate the g6k object.
    :param verbose: boolean indicating whether or not to output progress of BKZ.

    :returns: Siever object containing the reduced basis.
    """
    g6k = Siever(B, params)
    bkz = BKZReduction(g6k.M)

    # Run BKZ up to blocksize `beta`:
    for _beta in range(2, beta + 1):
        if verbose:
            print(f"\rBKZ_{_beta}", end="")
            stdout.flush()
        bkz(BKZ.Param(_beta, strategies=BKZ.DEFAULT_STRATEGY, max_loops=2))
    if verbose:
        print("\rBKZ reduction complete!")
        stdout.flush()
    return g6k


def progressive_sieve(g6k, l, r, verbose=False):
    """
    Sieve in [l, r) progressively. The g6k object will contain a list of short vectors.

    :param g6k: Siever object used for sieving
    :param l: integer indicating number of basis vectors to skip at the beginning. Taking l>0 gives a projected sublattice of the full basis.
    :param r: integer indicating up to where to sieve.
    :param verbose: boolean indicating whether or not to output progress of sieving.
    """
    if verbose:
        print("\rSieving", end="")
        stdout.flush()
    g6k.initialize_local(l, max(l, r - 20), r)
    g6k(alg="gauss")
    while g6k.l > l:
        # Perform progressive sieving with the `extend_left` operation
        if verbose:
            print("\rSieving [%3d, %3d]..." % (g6k.l, g6k.r), end="")
            stdout.flush()
        g6k.extend_left()
        g6k("bgj1" if g6k.r - g6k.l >= 45 else "gauss")
    with g6k.temp_params(saturation_ratio=.9, db_size_factor=6):
        g6k(alg="hk3")
    # Number of dual vectors that is used in a full sieve is (4/3)^{n/2}.
    g6k.resize_db(ceil(.9 * (4 / 3)**((r - l) / 2)))
    if verbose:
        print("\rSieving is complete!     ")
        stdout.flush()
    return g6k


def change_basis(basis, vector):
    """
    Puts a dual vector `vector` specified by coefficients into basis `basis`,
    i.e. calculate `vector * basis`.
    :param basis: an IntegerMatrix containing the basis for a lattice.
    :param vector: a vector specifying a lattice vector by its coefficients (in
                   terms of `basis`).
    :returns: the same lattice vector as `vector` but now expressed in the canonical basis.
    """
    return basis.multiply_left(vector)


def short_vectors_sampling(basis, threads, verbose):
    """
    Perform [Alg. 3, MATZOV].
    We take beta_2 = d and D = (4/3)^{d/2} so we get an output of a full sieve.
    As such we may take beta_1 = 3 since BKZ reduction does not change anything.
    We only perform one run so there is no need to randomize the basis, by [Remark 4.3, MATZOV].
    """
    print(f"Using {threads} threads")
    # If dual_mode=True, the dual (inverse transpose of B) is used implicitly.
    # Note that g6k object will still contain the primal basis.
    sieve_params = SieverParams(threads=threads, dual_mode=False)

    # 3: Run BKZ_{d,beta_1} on B to obtain a reduced basis g6k.M.B [Alg. 3, MATZOV].
    # g6k = progressive_BKZ(B_dual, (n + k_lat)//2, sieve_params, verbose=True)
    g6k = progressive_BKZ(basis, 3, sieve_params, verbose=verbose >= 1)

    # 4: Run a sieve of dimension beta_2 on the sublattice g6k.M.B to obtain a
    #    list of vectors, and add them to L [Alg. 3, MATZOV].
    progressive_sieve(g6k, 0, n + k_lat, verbose=verbose >= 0)

    # Initially, dual vectors are given by coefficients in terms of the basis B
    print("(1/4) Sieving dual vectors...")
    with Pool(threads) as pool:
        database = pool.map(partial(change_basis, g6k.M.B), g6k.itervalues())

    # 6: return L [Alg. 3, MATZOV].
    # Implicitly, G6K only stores one of the two vectors in {w, -w}, so give both explicitly here.
    return [w[:n] for w in database] + [[-x for x in w[:n]] for w in database]


def CDF_data(scores):
    """
    Return a list of the cumulative density function, assuming a list of
    samples is given in `scores`.  Only the size of the list matters, as it is
    assumed the scores are sorted when a plot of the scores is made with the
    score on the x-axis and the CDF on the y-axis.
    :param scores: A list of scores.
    """
    n = len(scores)
    return [i/n for i in range(1, n + 1)]


def plot_prediction(sp, data, avg, var, colour, legend):
    """
    Plots a prediction of a normal distribution in matplotlib.
    :param sp: the subplot in which to plot
    :param data: the x-values that are used, thus determining in which range for x the prediction should be plotted.
    :param avg: the expected mean of the normal distribution
    :param var: the expected variance of the normal distribution
    :param colour: the colour to use for the dashed line with prediction
    :param legend: a label for the prediction curve.
    """
    minx, maxx = min(data), max(data)
    xs = [minx + i * (maxx - minx) / 1000 for i in range(1001)]
    ys = [.5 + .5 * erf((x - avg) / sqrt(2 * var)) for x in xs]
    sp.plot(xs, ys, linestyle='dashed', color=colour, label=legend)


###############################################################################
if __name__ == '__main__':
    # Parse the command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='count', default=0, help='verbosity level')
    parser.add_argument('-j', type=int, default=cpu_count()//2, help='number of threads')

    # parser.add_argument('-n', type=int, default=40, help='Number of secret coefficients')
    parser.add_argument('--enum', type=int, default=0, help='Dimension of the enumeration step')
    parser.add_argument('--fft', type=int, default=10, help='Dimension of the FFT guessing step')
    parser.add_argument('--lat', type=int, default=10, help='Dimension of the remaining lattice')

    parser.add_argument('-q', type=int, default=3329, help='Prime to use in the q-ary lattice')
    parser.add_argument('-p', type=int, default=5, help='Prime to use in modulus switching')
    parser.add_argument('--eta', type=int, default=3, help='Eta for binomial distribution')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    args = parser.parse_args()

    threads = min(cpu_count(), args.j)
    n = args.enum + args.fft + args.lat
    q = args.q
    p = args.p

    # Beware:
    # MATZOV uses column notation, but G6K uses row notation.
    # Primal basis, in LWE form
    A, B_primal = generate_LWE_lattice(2 * n, n, q)

    # Split A = (A_enum | A_fft | A_lat) according to [p.14,MATZOV]
    k_enum, k_fft, k_lat = args.enum, args.fft, args.lat
    assert k_enum >= 0 and k_fft >= 0 and k_lat >= 0

    # 1: Compute A_lat composed of the last k_lat columns of A [Alg. 2, MATZOV].
    A_enum, A_fft, A_lat = A[:k_enum], A[k_enum:n - k_lat], A[n - k_lat:]

    # Transpose the parts of A
    A_enum.transpose()
    A_fft.transpose()
    A_lat.transpose()

    # With column notation, let the primal basis be:
    # [ qI_n -A_lat ]
    # [  0   I_klat ]
    # Then the dual is, in column notation, given as:
    # [ I_n      0       ]
    # [ A_lat^T  qI_klat ]

    # 2: Compute the matrix B = [[I_m, 0 ], [A_lat^T, qI_{k_lat}]], where alpha = sigma_e/sigma_s
    # [Alg. 2, MATZOV]. Here, we assume chi_s = chi_e so alpha = 1.
    B_dual = IntegerMatrix.identity(n + k_lat)
    for i in range(n, n + k_lat):
        B_dual[i, i] *= q
    for i in range(0, n):
        for j in range(0, k_lat):
            B_dual[i, n + j] = A_lat[i, j] % q

    # 3: Run the short vectors sampling algorithm (Algorithm 3) on the basis B
    #    with parameters beta_1, beta_2, D, to get a list L of D short vectors
    #    [Alg. 2, MATZOV].
    L = short_vectors_sampling(B_dual, threads, args.v)
    N = len(L)
    print(f"Database contains {N} dual vectors")

    # Perform Line 7 and 8 as these do not depend on the guess of s_enum:
    with Pool(threads) as pool:
        print("(2/4) Computing y_fft...")
        # 7: Compute y_{j,fft} = x_j^T A_{fft} [Alg. 2, MATZOV].
        y_ffts = pool.map(partial(change_basis, A_fft), L)

        print("(3/4) Computing y_enum...")
        # 8: Compute y_{j,enum} = x_j^T A_{enum} [Alg. 2, MATZOV].
        y_enums = pool.map(partial(change_basis, A_enum), L)

    samples = args.samples
    # Compute the score for the secret:
    secret_dist = CentredBinomial(args.eta)
    print(f"(4/4) Calculating {samples} scores with mod switching for p={p}...")

    def work(_):
        """
        Compute the score for the correct guess given a randomly generated
        target, with the way described in [MATZOV]. We only calculate FFT(T) at
        the (correct) index `stilde_fft (mod p)`, which does not require the
        (slower) FFT.
        :param _: unused index variable
        :returns: a tuple containing:
            1) the score without modulus switching,
            2) the score with modulus switching for all the primes `ps`,
            3) the squared norm of s_lat and error, used to determine \\tau in [MATZOV, Lemma 5.3].
        """
        # Generate the secret:
        secret = [secret_dist() for i in range(n)]
        error = [secret_dist() for i in range(n)]
        # The target is called `b` in MATZOV, i.e. target = A * secret + error.
        target = np.add(A.multiply_left(secret), error)
        # Assume we take the correct guess for both s_enum, and s_fft (mod q):

        # 5: Initialize a table T of dimensions p x p x ... x p (k_fft times) [Alg. 2, MATZOV].
        table = np.zeros(shape=(p,) * k_fft, dtype=complex)
        # ms_score = 0.0

        # 6: for every short vector (alpha x_j, y_lat) in L do [Alg. 2, MATZOV].
        # Note: y_lat is unused in Alg. 2 so this is not stored.
        for (j, x_j) in enumerate(L):
            # Determine the score when using modulus switching:
            # 9: Add e^{(x_j^T b - y_{j,enum}^T s_enum)*2pi i/q} to cell
            # [ p/q y_{j,fft} ] of T [Alg. 2, MATZOV].
            index = tuple(round(p * x / q) % p for x in y_ffts[j])
            inner_product = np.inner(x_j, target) - np.inner(y_enums[j], secret[:k_enum])
            angle = inner_product * 2 * pi / q
            table[index] += cos(angle) + sin(angle) * 1.j
            # ms_score += cos(inner_product * 2 * pi / q - np.inner(index, secret[k_enum:k_enum+k_fft]) * 2 * pi / p)
            # Because the dual database is symmetric, the score is always real, so use cos().
        # 10: end for [Alg. 2, MATZOV].

        # 11: Perform FFT on T [Alg. 2, MATZOV].
        fft_output = np.fft.fftn(table).real

        # In practice, this maximum is attained at the right guess of s_fft (mod p).
        ms_score = fft_output.max()

        # sfft_index = sum((secret[k_enum + i]%p) * p**(k_fft-1-i) for i in range(k_fft))
        # _wr_scores = fft_output.flatten()
        # print(_wr_scores[sfft_index], " vs ", ms_score)
        # assert _wr_scores[sfft_index] == ms_score
        # _wr_scores = np.delete(_wr_scores, sfft_index)
        # _wr_scores.sort()

        square_error = np.inner(secret[n-k_lat:], secret[n-k_lat:]) + np.inner(error, error)
        # return ms_score, _wr_scores, square_error
        return ms_score, square_error

    ms_scores, wr_scores, avg_sq_error = [], [], 0
    with Pool(threads) as pool:
        # for (_ms_score, __wr_scores, _sq_err) in pool.imap_unordered(work, range(samples)):
        for (_ms_score, _sq_err) in pool.imap_unordered(work, range(samples)):
            ms_scores.append(_ms_score)
            # wr_scores.append(__wr_scores)
            avg_sq_error += _sq_err

    # Sort the scores
    ms_scores.sort()
    # wr_scores = np.concatenate(wr_scores)
    # wr_scores.sort()

    # print(ms_scores)
    # print(wr_scores)

    # Collect expectations and variances
    avg_score = sum(ms_scores) / samples
    var_score = sum(x*x for x in ms_scores) / samples - avg_score**2

    avg_sq_error /= samples
    # Now avg_error_sq equals E[ ||e||^2 + ||s_{lat}||^2 ], as used in \\tau~[MATZOV, Lemma 5.3].

    exp_avg_err = variance(secret_dist) * (k_lat + n)
    print(f"Variance: {variance(secret_dist)}")
    print(f"Average error={avg_sq_error}, expected={exp_avg_err}")

    # Give expectations on the quantity, i.e.
    # - D_eq is in Sect. 5.2 [MATZOV], with \ell = \sqrt(4/3) * GH(n + k_lat) * det(B_dual)
    # - \widetilde{D_round} is in Thm. 5.9 [MATZOV], however we need D_round^{-1/2}.
    exp_ell = sqrt(4/3) * GH(n + k_lat) * q**(k_lat / (n + k_lat))
    # In fact, this is a lower bound:
    exp_eps_eq = exp(-2 * pi**2 * avg_sq_error * exp_ell**2 / ((n + k_lat) * q**2))
    # In fact, this is an upper bound:
    var_eps_eq = 0.5 + exp_eps_eq**4

    exp_eps_round = prod((sin(pi*x/p) / (pi*x/p))**(secret_dist.PDF(x) * k_fft) for x in secret_dist.support() if x%p != 0)
    exp_mean = exp_eps_eq * exp_eps_round * N
    exp_var = var_eps_eq * N

    print()
    print(f"n={n}, k_enum={k_enum}, k_fft={k_fft}, k_lat={k_lat}, q={q}, samples={samples}, eta={args.eta}")
    print("*** NO MODULUS SWITCHING ***")
    print(f"Expected score = {exp_eps_eq * N:.2f} +/- {sqrt(exp_var):.2f}")

    print()
    print("*** MODULUS SWITCHING ***")
    print(f"Expected score = {exp_eps_eq * exp_eps_round * N:.2f} +/- {sqrt(exp_var):.2f}")
    print(f"Actual   score = {avg_score:.2f} +/- {sqrt(var_score):.2f}")

    fig = plt.figure()
    sub_plot = fig.add_subplot(1, 1, 1)

    # These are max score so don't plot it.
    # sub_plot.plot(scores, CDF_data(scores), color='tab:red', label='no MS')
    # plot_prediction(sub_plot, scores, exp_eps_eq * N, exp_var, 'tab:red', 'pred no MS')

    print("p, pred. mean, pred. median, pred. sigma, meas. mean, meas. median, meas. sigma")
    print(f"{p}, {exp_mean:.2e}, {exp_mean:.2e}, {sqrt(exp_var):.0f}, {avg_score:.2e}, {median(ms_scores):.2e}, {sqrt(var_score):.2e}")

    sub_plot.plot(ms_scores, CDF_data(ms_scores), color='tab:blue', label=f'MS({p})')
    plot_prediction(sub_plot, ms_scores, exp_mean, exp_var, 'tab:blue', f'pred MS({p})')

    print()
    plt.legend()
    plt.title(f"Score distributions for n={n},k_enum={k_enum},k_fft={k_fft},k_lat={k_lat}, #samples={samples}")
    plt.savefig(f"mod_switch_{n}_{k_enum}_{k_fft}_{samples}.png", dpi=300)
    plt.show()
