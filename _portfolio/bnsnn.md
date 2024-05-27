---
title: "CinchNET"
excerpt: "GNN that message passes features within a simplex, essentially bottlenecking it within a simplex"
collection: portfolio
---

We introduce a novel graph neural network architecture where messages are bottlenecked mostly within a simplex itself. This allows messages to be passed in between nodes that are part of a simplex, increasing node classification probability. The assumption one needs to make is that the simplex itself has a semantic meaning. The main file may be found [here](/files/GSN.py). As for the preprocessing step, we encode a [functor](/files/AdjFunctor.py) that unrolls a simplicial set back into a graph.
