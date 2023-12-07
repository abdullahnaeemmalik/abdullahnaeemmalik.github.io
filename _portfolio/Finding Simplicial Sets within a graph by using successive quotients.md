---
title: "Finding Simplicial Sets within a graph by using successive quotients"
excerpt: "This Jupyter notebook encodes a solution to the problem of finding all n-simplices within a graph."
collection: portfolio
---

The notebook may be found [here](/files/Graph Towers Approach DB Focus.ipynb). The code is tested with random graphs and with transitive tournaments. This is accomplished by taking successive quotients of the graph, and then lifting up simplices to the pre-image of each quotient. The idea mimics HNSW architecture. All successive quotients are saved in a databse using SQL for an organized and easier access
