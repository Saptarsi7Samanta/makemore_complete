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

```
8 characters -> FlattenConsecutive(2) -> Linear -> BatchNorm -> Tanh
4 characters -> FlattenConsecutive(2) -> Linear -> BatchNorm -> Tanh
2 characters -> FlattenConsecutive(2) -> Linear -> BatchNorm
1 output     -> Linear -> logits
```

This lets the network build up representations gradually across a longer context (8 characters), rather than combining everything in one flat step.

**3. Results**
- 176,875 parameters
- Train loss: **1.928** | Validation loss: **2.041**
- Improvement over the earlier MLP version's validation loss of 2.095

**4. BatchNorm folding (extra exercise)**
Implemented and verified that a trained BatchNorm layer's `gamma`/`beta` parameters can be mathematically folded directly into the preceding `Linear` layer's weights and bias — collapsing two operations into one. Verified with `torch.allclose()` that the folded model produces numerically identical output to the original, confirming BatchNorm is purely a training-time stabilization technique that can be discarded at inference once folded.

## Tech
Python, PyTorch, Matplotlib

## Credits
[Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) — Building a WaveNet
