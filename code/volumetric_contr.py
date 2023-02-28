from math import sqrt, e, exp, log, pi

# c0 is a tweaking constant, we simply take equal to 1. It is predicted that we need `c0 * log(T)
# <= N epsilon^2` for the attack to work. However, changing c0 to a value just above 1 makes one
# more confident that it should work, while c0 < 1 may give weird results: the attack shouldn't
# work in this regime.
c0 = 1.


def gh(n):
    """
    Return the Gaussian Heuristic at dimension n. This gives a prediction of the length of the
    shortest vector in a lattice of unit volume.
    :param n: number indicating the dimension
    :returns: GH(n)
    """
    return sqrt(n * (pi * n)**(1.0/n) / (2 * pi * e))


def assert_GJ(n, logT, sigma):
    """
    Sanity check whether given a sigma, and number of targets T, the attack works.
    """
    N = (4./3)**(n/2.)
    ell = gh(n) * sqrt(4./3)

    epsilon = exp(-2 * pi**2 * sigma**2 * ell**2)
    # N_ equals the number of dual vectors needed for the attack to work.
    N_ = c0 * logT / epsilon**2
    # Assert that N is smaller than 1.01 N_
    assert N <= 1.0001 * N_


def sigma_GJ(n, logT):
    """
    Compute the sigma up to which the Guo-Johansson dual attack of distinguishing a BDD sample with
    parameter sigma from T many uniform samples, should still work with probability close to 1, for
    a lattice in dimension `n` of unit volume.
    :param n: the dimension
    :param logT: the natural log of the number of uniform targets that is taken.
    :returns: max. sigma up to which distinguishing the BDD with parameter sigma still works.
    """

    # Number of dual vectors from a full sieve equals `(4/3)^{n/2}`.
    N = (4./3)**(n/2.)
    # The dual vectors have a length concentrated around `sqrt(4/3)*GH(n)`.
    ell = gh(n) * sqrt(4./3)
    # The minimal advantage that we must have to have the attack work, must satisfy:
    # `log(T) <= N epsilon^2.
    epsilon = sqrt(c0 * logT / N)
    # We have `epsilon = exp(-2pi^2 sigma^2 ell^2)`, giving an upper bound on sigma.
    sigma = sqrt(.5 * log(1.0/epsilon)) / (pi * ell)
    return sigma


def sigma_vol(n, logT):
    """
    Given a number of targets T and a dimension `n`, we expect with constant probability that one
    out of T uniform targets will have a distance `r` to a lattice point, if `r` is taken large
    enough. Assuming a random lattice of volume 1, we return `r/√n` for the smallest `r` still
    giving a constant probability, i.e. r = 1/T^{1/n}.
    """
    r = gh(n) * exp(-logT / n)
    return r / sqrt(n)


def do_dim(n):
    for i in range(10, 3000):
        lgT = 0.1*i
        logT = log(2) * lgT

        try:
            # Determine the maximal sigma for which the dual attack is expected to work
            sigma_BDD = sigma_GJ(n, logT)
        except ValueError:
            continue
        assert_GJ(n, logT, sigma_BDD)
        # Determine the sigma for which, with this number of targets we expect to find a uniform
        # sample at radius sigma_UNIF √n.
        sigma_UNIF = sigma_vol(n, logT)

        if sigma_BDD > sigma_UNIF:
            # Now, we have (with const. prob.) a uniform sample with a distance closer to the
            # lattice than the BDD sample with parameter sigma=sigma_BDD, that is a contradiction!
            ghf_BDD = sigma_BDD * sqrt(n) / gh(n)
            ghf_UNIF = sigma_UNIF * sqrt(n) / gh(n)
            print(f"{n:4d}, {lgT:8.3f}, {lgT/n:10.3f}, {ghf_BDD:7.3f}, {ghf_UNIF:8.3f}")
            break


print("   n, log_2(T), log_2(T)/n, ghf_BDD, ghf_UNIF")
for n in range(30, 100):
    do_dim(n)
for n in range(100, 501, 5):
    do_dim(n)
