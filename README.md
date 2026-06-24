# P.rigidum-inspired Channel Pruning
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

PyTorch implementation of biologically inspired structured channel pruning for CNNs, drawing on the adaptive network optimization mechanism of *Physarum rigidum*. The method evaluates channel importance via cosine dissimilarity between input and output feature maps, pruning redundant pathways while preserving activation flow coherence.

## Highlights
- Lightweight metric computed in one single forward pass, no auxiliary training required
- True structured pruning for direct inference acceleration and memory reduction
- Universal core module adaptable to any CNN architecture with convolutional layers

## Installation
```bash
git clone https://github.com/YueyingShi0315/P.rigidum_inspired_pruning.git
cd P.rigidum_inspired_pruning
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

## Quick Start
### Core pruning API
```python
from p_rigidum_inspired_pruning import p_rigidum_prune_indices, apply_pruning

# Get indices of channels to retain
keep_masks = p_rigidum_prune_indices(model, test_loader, prune_ratio=0.4)
# Generate the pruned model
pruned_model = apply_pruning(model, keep_masks)
```
A short fine-tuning phase after pruning is recommended to recover model accuracy.

### Full demo
An end-to-end workflow including baseline training, pruning, fine-tuning and evaluation on MNIST is available in `p_rigidum_inspired_pruning_demo.ipynb`.

## Repository Structure
| File | Description |
|:-----|:------------|
| `p_rigidum_inspired_pruning.py` | Core implementation of the pruning algorithm |
| `p_rigidum_inspired_pruning_demo.ipynb` | Interactive Jupyter notebook demo |

## Citation
If you use this code in your research, please cite:
```bibtex
@misc{shi2026pruning,
  title = {Physarum rigidum-inspired Structured Channel Pruning for Convolutional Neural Networks},
  author = {Shi, Yueying},
  year = {2026},
  url = {https://github.com/YueyingShi0315/P.rigidum_inspired_pruning}
}
```

## License
This project is released under the MIT License.
