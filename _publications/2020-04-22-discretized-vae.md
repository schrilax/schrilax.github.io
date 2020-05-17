---
title: "Discretized Bottleneck in VAE: Posterior-Collapse-Free Sequence-to-Sequence Learning"
collection: publications
permalink: /publication/2020-04-22-discretized-vae
date: 2020-04-22
venue: 'Conference (under submission)'
paperurl: 'https://arxiv.org/abs/2004.10603'
citation: 'Discretized Bottleneck in VAE: Posterior-Collapse-Free Sequence-to-Sequence Learning
Y Zhao, P Yu, S Mahapatra, Q Su, C Chen - arXiv preprint arXiv:2004.10603, 2020.'
---
Variational autoencoders (VAEs) are important tools in end-to-end representation learning. VAEs can capture complex data distributions and have been applied extensively in many NLP tasks. However, a common pitfall in sequence-to-sequence learning with VAEs is the posterior-collapse issue in latent space, wherein the model tends to ignore latent variables when a strong auto-regressive decoder is implemented. In this paper, we propose a principled approach to eliminate this issue by applying a discretized bottleneck in the latent space. Specifically, we impose a shared discrete latent space where each input is learned to choose a combination of shared latent atoms as its latent representation. Compared with VAEs employing continuous latent variables, our model endows more promising capability in modeling underlying semantics of discrete sequences and can thus provide more interpretative latent structures. Empirically, we demonstrate the efficiency and effectiveness of our model on a broad range of tasks, including language modeling, unaligned text style transfer, dialog response generation, and neural machine translation.