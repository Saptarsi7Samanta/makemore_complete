# Makemore WaveNet — Hierarchical Character-Level Language Model

A character-level language model implementing a WaveNet-inspired hierarchical architecture, built while completing Andrej Karpathy's "Neural Networks: Zero to Hero" series. This is the final and most advanced version of the Makemore project — building on the bigram, trigram, and MLP+embeddings versions that came before it.

## What's included

**1. A custom PyTorch-style module system**
Built from scratch, mirroring how real PyTorch layers are structured internally:
- `Linear` — a standard fully-connected layer
- `BatchNorm1d` — batch normalization with running mean/variance for inference
- `Tanh` — nonlinearity
- `Embedding` — learned character embeddings
- `FlattenConsecutive` — merges pairs of adjacent time-steps, the key building block of the hierarchical structure
- `Sequential` — chains layers together into a full model

**2. Hierarchical (WaveNet-style) architecture**
Unlike a standard MLP that flattens all context characters into the network at once, this progressively merges context in stages:
