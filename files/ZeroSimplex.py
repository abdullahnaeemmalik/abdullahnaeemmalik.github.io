class ZeroSimplicialSet:
    def __init__(self, name, point):
        self.point = point
        self.name = [self.point]

    def simplicialset(self, num):
        if num == 0:
            return(self.name)
        X=self.name
        for i in range(num):
            X = X + [self.point]
        return(X)

    def facemap(self, num):
        return f"d({self.simplicialset(num)}) = {self.simplicialset(num-1)}"

    def degeneracymap(self, num):
        return f"s({self.simplicialset(num)}) = {self.simplicialset(num+1)}"
