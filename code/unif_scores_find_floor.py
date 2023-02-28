import os

gap = 1

print("  n, T_floor")
for n in range(120):
    filename = f"../data/unif_scores_n{n}.csv"
    if not os.path.isfile(filename):
        continue
    with open(filename, "r") as file:
        res = None
        for line in file:
            try:
                score, predp, realp = list(map(float, line.split(',')[:3]))
            except ValueError:
                continue
            if predp < realp - gap:
                res = -realp
                break
                
        if res is not None:
            print(f"{n:3}, {res:.3f}")
