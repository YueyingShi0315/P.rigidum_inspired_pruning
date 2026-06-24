# P.rigidum-inspired Channel Pruning
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

PyTorch implementation of biologically inspired structured channel pruning for CNNs, drawing on the adaptive network optimization mechanism of *Physarum rigidum*. The method evaluates channel importance via cosine dissimilarity between input and output feature maps, pruning redundant pathways while preserving activation flow coherence.
