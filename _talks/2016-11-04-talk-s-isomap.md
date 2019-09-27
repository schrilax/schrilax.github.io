---
title: "Error Metrics for Learning Reliable Manifolds from Streaming Data"
collection: talks
type: "Talk"
permalink: /talks/2016-11-04-talk-s-isomap
venue: "UB Department of Computer Science and Engineering"
date: 2016-11-04
location: "Buffalo, New York"
---

Spectral dimensionality reduction is frequently used to identify low-dimensional structure in high-dimensional data. However, learning manifolds, especially from the streaming data, is computationally and memory expensive. In this paper, we argue that a stable manifold can be learned using only a fraction of the stream, and the remaining stream can be mapped to the manifold in a significantly less costly manner. Identifying the transition point at which the manifold is stable is the key step. We present error metrics that allow us to identify the transition point for a given stream by quantitatively assessing the quality of a manifold learned using Isomap. We further propose an efficient mapping algorithm, called S-Isomap, that can be used to map new samples onto the stable manifold. We describe experiments on a variety of data sets that show that the proposed approach is computationally efficient without sacrificing accuracy.

[[Poster](https://schrilax.github.io/files/Mixer16.pdf)]