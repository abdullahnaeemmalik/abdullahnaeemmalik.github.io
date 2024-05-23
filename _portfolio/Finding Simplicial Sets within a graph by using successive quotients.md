---
title: "Finding Simplicial Sets within a graph by using successive quotients"
excerpt: "This Jupyter notebook encodes a solution to the problem of finding all n-simplices within a graph."
collection: portfolio
---

The notebook may be found [here](/files/Graph Towers Approach DB Focus.ipynb). The code is tested with random graphs and with transitive tournaments. This is accomplished by taking successive quotients of the graph, and then lifting up simplices to the pre-image of each quotient. The idea mimics HNSW architecture. All successive quotients are saved in a databse using SQL for an organized and easier access. The runtime for this algorithm is linear in edges and vertices. 

As of now, this algorithm works for a small ratio of collapse -- we want to make sure that the not too many simplices get collapsed at the same time! To correct this, one could add in a function that keeps track of the longest length of edges collapsed, and then modify the procedure for the lift accordingly.
