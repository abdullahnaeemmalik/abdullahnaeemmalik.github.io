#take, for example, the following set of vertices:
V = [1,2,3]

def X(n):
    if n == 0:
        return(V)
    X = [[v] for v in V]
    for i in range(n):
        X = [r + [v] for r in X for v in V if r[-1] <= v]
    return(X)

X(2)
#returns all 2-simplices, including degenerate ones.
