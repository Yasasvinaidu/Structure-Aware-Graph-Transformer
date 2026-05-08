# Structure-Aware Transformer for Graph Representation Learning

This project implements and analyzes the Structure-Aware Transformer (SAT) model
for graph representation learning.

## Paper
https://arxiv.org/abs/2202.03036

## Report
https://www.overleaf.com/read/bxsfsnzjbzkd#9063b7

## Acknowledgements and Attribution
The foundational codebase for the Structure-Aware Transformer (SAT) was originally developed by Chen et al. and is available at the [BorgwardtLab/SAT repository](https://github.com/BorgwardtLab/SAT). 

Our work forks and builds upon this original PyTorch implementation. Specifically, the base extraction mechanisms in the `/sat` and `/experiments` directories are adapted from the original authors. All novel architectural extensions (GAT ablation, MPNN edge-conditioning, Gated Residuals, Adaptive Positional Encodings, and Hierarchical Chunking) documented in `/extensions/extensions.md` are original contributions developed for this course project.
