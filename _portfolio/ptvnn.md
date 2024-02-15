---
title: "PseudoTop Vertex Neural Network"
excerpt: "GNN that message passes features of pseudotop vertices, after a refinement"
collection: portfolio
---

We introduce a novel graph neural network architecture. The message passing is based on not only the information granted by a node vector, but also incorporates the place of a vertex as a pseudo-top vertex. Pseudo top vertices generalize the notion of a directed clique, but only at the level of a node, so global information is stored locally. This allows the implementation of message passing without the need for new machine learning tools. For details on its effectiveness, rationale, and how the architecture respects the variance-bias tradeoff, please see the related talk [here](files/ptvnn.pdf). The source code for [main](/files/pTV.py), [modified GCN](/files/GCN_modified.py) and [utilities](/files/utils.py) can be found by clicking on the relevant links. The class for the preprocessing step is available [here](/files/ptv_finder.py). 
