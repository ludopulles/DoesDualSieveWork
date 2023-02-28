# Analysis of the Dual Attack
This repository contains:
* All the data that are used to plot the figures and generate the tables in the accompanying [paper](https://eprint.iacr.org/2023/302).
* A script `code/unif_scores.py` that determines the experimental score distribution for uniform targets.
* A file `code/dual_utils.c`, written in C, that is used in `code/unif_scores.py` to perform a quick Walsh--Hadamard Transform.
* A Makefile `code/Makefile` that is used to compile the `code/dual_utils.c` into a shared library that can be used by the Python file `code/unif_scores.py`. Run `make` in the directory `code` to build `code/dual_utils.so`.
* A script `code/bdd_scores.py` that determines the experimental score distribution for a BDD target with a given GH-factor.
* A script `code/mod_switch.py` that determines the score distribution for the correct guess (s_enum, s_fft) in case of modulus switching as described in [MATZOV](https://zenodo.org/record/6493704).
* A script `code/volumetric_contr.py` that determines the Contradictory Regime, i.e. the data in Figure 2.
* A script `code/unif_scores_find_floor.py` that determines the point(s) where the "floor" phenomenon starts to kick in, based on the data files `data/unif_scores_n*.csv`.
* A script `code/guess_time.py` that calculates the entropy and the expected number of enumerations in the MATZOV attack (Appendix A.4).

## Requirements
* Preferably, a UNIX machine. On windows, you may perhaps have to change `dual_utils.so` to `dual_utils.dll` or so.
* C compiler, preferably `gcc`
* Python3
* [G6K](https://github.com/fplll/g6k/) which can be installed by running `pip install g6k`.
