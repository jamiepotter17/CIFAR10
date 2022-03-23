# CIFAR-10 Image Classifier

### A Project for Udacity's 'Into to Machine Learning with Pytorch' Nanodegree

---

## Problem Statement

This image classifier is trained and evaluated on one of the most well-studied computer vision data sets in existence - CIFAR-10. The scenario is that we have to imagine ourselves inside a fictional company. Management are trying to decide whether to build or buy an object detection algorithm for objects that may be on the side of the road. They are considering buying a computer vision algorithm from a company called Detectocorp. Detectocorp’s algorithm claims a 70% accuracy rate on the CIFAR-10 dataset, a benchmark used to evaluate the state of the art for computer vision systems.

But before making this purchase, management wants to explore whether you can build an in-house solution that performs well. They have asked you to try your hand at creating a neural network that can classify arbitrary objects and potentially be fine-tuned on a larger dataset using transfer learning.

Using the skills you’ve acquired building neural networks using PyTorch, your task is to build an image classifier using the CIFAR-10 dataset and evaluate its accuracy. Then you'll compare its performance to both Detectocorp’s algorithm (which achieved 70% accuracy) as well as the state of the art results detailed in the notebook—and make a recommendation to management about whether to build the solution in-house or buy the algorithm from Detectocorp.

## Implementation

See 'CIFAR-10_Image_Classifier-STARTER.ipynb' for details, but in essence I created my own convolutional neural network by starting from the Resnet50 model and cutting off the final classification layer, replacing it with:

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                       nn.ReLU(),
                       nn.Linear(512, 128),
                       nn.ReLU(),
                       nn.Linear(128, 32),
                       nn.ReLU(),
                       nn.Linear(32, 10),
                       nn.LogSoftmax(dim=1))

## Results and Recommendation

The CNN achieved an accuracy score of 67.8%, only slightly worse than the claimed 70% of Detectocorp. However, given that scores of over 90% have been achieved, e.g. 96% by Zagoruyko et al. in 2016, it seems likely that better results could be achieved by hiring an outside consultant. It may also be possible that we could achieve better ourselves by adopting the wide Resnet model used by Zagoruyko et al and training for a long period of time, if the time and resources are there.
