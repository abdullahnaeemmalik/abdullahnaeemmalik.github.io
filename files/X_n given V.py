V = [1, 2, 4]

def X(n):
    if n == 0:
        return(V)
    X = [[v] for v in V]
    for i in range(n):
        X = [r + [v] for r in X for v in V if r[-1] <= v]
    return(X)

return(X(2))
