---
title: "S-Isomap++: Multi manifold learning from streaming data"
collection: talks
type: "Talk"
permalink: /talks/2017-12-12-talk-s-isomap-pp
venue: "2017 IEEE International Conference on Big Data"
date: 2017-12-12
location: "Boston, Massachusetts"
---

Manifold learning based methods have been widely used for non-linear dimensionality reduction (NLDR). However, in many practical settings, the need to process streaming data is a challenge for such methods, owing to the high computational complexity involved. Moreover, most methods operate under the assumption that the input data is sampled from a single manifold, embedded in a high dimensional space. We propose a method for streaming NLDR when the observed data is either sampled from multiple manifolds or irregularly sampled from a single manifold. We show that existing NLDR methods, such as Isomap, fail in such situations, primarily because they rely on smoothness and continuity of the underlying manifold, which is violated in the scenarios explored in this paper. However, the proposed algorithm is able to learn effectively in presence of multiple, and potentially intersecting, manifolds, while allowing for the input data to arrive as a massive stream.