# MIFCN



Optical Coherence Tomography Image Denoising via Multi-input Fully-Convolutional Networks [paper link](https://doi.org/10.1016/j.compbiomed.2019.01.010)

The tensorflow implementation of the proposed MIFCN method.



## Results

The results are stored [here](https://github.com/ashkan-abbasi66/MIFCN/tree/master/RESULTS).



## Training dataset

The 15x15 patch pairs that were extracted from noisy and clean image pairs were saved in two numpy files:

- `train15_inputs.npy`: It contains a numpy array with size [4000, 5, 15, 15]. The array contains 400 patches with size 15x15 that were extracted from 10 noisy images. For each patch, we have extracted 4 additional patches by nonlocal searching.
- `train15_labels.npy`: It contains the corresponding clean patches.

**Loading the training dataset:**

```python
import numpy as np
inputs_ndarray = np.load('data/train15_inputs.npy')
labels_ndarray = np.load('data/train15_labels.npy')
```

### How was the training dataset collected?

The following is a full description of how we collect the training patches.

To train and evaluate our proposed MIFCN method, we have used the SDOCT datasets that were made
publicly available by [Fang et al](http://people.duke.edu/~sf59/Fang_TMI_2013.htm) [1]. In the training part, there are 10 high SNR (almost clean) images with their corresponding low SNR (noisy) images. However, in the test part, we not only have a pair of noisy and clean images but also there are four additional noisy OCT images. These noisy images were captured from spatially very close positions. We call them nearby OCT images.

Since in the training set, we do not have nearby OCT images and their corresponding clean images, we extract N=400 patches of size 15x15 pixels from each noisy and clean image pair. Then, for each patch, we extracted 5 most similar patches (including the patch itself). We have used the nonlocal searching [19] procedure for finding similar patches for each patch in an image. Note that the patch and its similar ones belong to the same image. 

**1) Cropping the original training images**

Since there are a large background portion in OCT images, we manually cropped a portion containing the retina from each training image pairs. 

**2) Patch Extraction**

We extracted 15x15 patches from each cropped image pairs.

**3) Save the extracted patches in numpy files**

Since reading extracted patches as individual files from a hard disk is time-consuming, we saved all those patches in two numpy files: `train15_inputs.npy` and `train15_labels.npy`.

