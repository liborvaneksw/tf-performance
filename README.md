# tf-performance
Useful scripts for evaluating performance of different TensorFlow operations.

# Model Loading
If you are interested in learning how long does it take to load a model in different formats, 
please check [`model_loading.py`](model_loading.py). By default, it measures the loading times
of MobileNet V2, DenseNet201 and Resnet152 V2 with ImageNet weights saved in
five different ways: 
- SavedModel: load full model and weights,
- HDF5: load full model and weights,
- SavedModel: create model and load weights,
- HDF5:  create model and load weights and
- TF Lite: load full model and weights.

You only need to install **TensorFlow 2+** for this script.