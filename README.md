This repository for the compression of three Convolutional Neural Network models—Lenet, Alexnet, and VGG—for classifying marine animal datasets. The compressed models presented in this project not only maintain the same accuracy as the original models but also significantly reduce their memory foot-print. This work was implemented for **Deep learning** final project at NYU Tandon 2024.
# Authors:
- Raksi Kopo (rk4585@nyu.edu)
- Duc Anh Van (dv2223@nyu.edu)
- Remon Roshdy (rr3531@nyu.edu)
# Deep Compression
Compressing Convolutional Neural Network models with pruning, trained quantization
It provides an implementation of the two core methods:

- Pruning
- Quantization

These are the main results on the [Sea animal](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste) datasets

| Network       | ACC      | Compression Rate (Ours) | Compression Rate (Han et al.) |
|--------------------------|--------------------|--------------------------|-------------------------|-------------------------------|
| Lenet5        | 2.0%               | 1.64%                    | -                       | -                             |
| Alexnet       | 1.8%               | 1.58%                    | **48X**                 | 40X                           |
| VGG16            | 0.83%              | 0.8%                     | -                       | -                             |

## Requirements
  - pytorch
  - pytorch-lightning
  - torchmetrics
  - torchvision
  - ipykernel
  - jupyter
  - matplotlib
  - numpy
  - scipy
  - scikit-learn
  - tqdm
  - tensorboard

## Project Structure
  ```
  model-compression/
  │
  ├── notbook - main script to run
  ├── Data/ directory containing 14,000 sea animals images of 23 species 
  │   
## TODOs
- [ ] Run on Jupiter 

## References
[[1]](https://arxiv.org/pdf/1510.00149v5.pdf) Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." arXiv preprint arXiv:1510.00149 (2015)


## Acknowledgments
- [Pytorch](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils) for pruning and quantization library
- [Datasets](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste) for sea animal data
