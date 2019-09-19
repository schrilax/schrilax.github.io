---
title: "S-Isomap++: Multi Manifold Learning from Streaming Data"
collection: publications
permalink: /publications/2018-01-15-s-isomap-pp
date: 2018-01-15
venue: '2017 IEEE International Conference on Big Data (Big Data)'
paperurl: 'http://ieeexplore.ieee.org/document/8257987/'
citation: 'Mahapatra, Suchismit, and Varun Chandola. "S-Isomap++: Multi manifold learning from streaming data." In 2017 IEEE International Conference on Big Data (Big Data), pp. 716-725. IEEE, 2017.'
---
Manifold learning based methods have been widely used for non-linear dimensionality reduction (NLDR). However, in many practical settings, the need to process streaming data is a challenge for such methods, owing to the high computational complexity involved. Moreover, most methods operate under the assumption that the input data is sampled from a single manifold, embedded in a high dimensional space. We propose a method for streaming NLDR when the observed data is either sampled from multiple manifolds or irregularly sampled from a single manifold. We show that existing NLDR methods, such as Isomap, fail in such situations, primarily because they rely on smoothness and continuity of the underlying manifold, which is violated in the scenarios explored in this paper. However, the proposed algorithm is able to learn effectively in presence of multiple, and potentially intersecting, manifolds, while allowing for the input data to arrive as a massive stream.