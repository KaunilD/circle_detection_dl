#### CV Practical: Circle detection in a noisy image.

__Overview__

Goal of this assignment is to build a deep learning pipeline to predict parameters of a circle ( center (x, y) and radius (r)) embedded in a noisy/occluded image $I_N$. The pipeline delineated in the following sectionscontains 2 key elements:

1. The denoising backbone - DnCNN.
2. The feature extractor - CDNet.

Time taken for modelling the architecture: $\approx$ 20 minutes and cumulative training time for both the networks: $\approx$ 2 hours, spread out over a day on an AWS P2 instance.

__The denoising backbone: [DnCNN](https://arxiv.org/pdf/1608.03981.pdf)__

Inspired from the work by Kai Zhang et. al, the denoising backbone does most of the heavy lifting for CDNet. The denoising network implemented below is a simplified version of the original implementation preserving it's key aspects:
1. Batch normalization 
2. Residual Learning.

__N.B.__ While training DnCNN, the ride through the cost valley was smooth and mostly linear, however in case of CDNet the optimizer required a lot of navigating and guiding in form of model restarts and changing learning rate. As a result loading the DnCNN backbone weights using `CDNet._init_bbone()` should be carried out only once while the first training run.

##### Requirements:

A `requirements.txt` of my conda environment included.

1. [pytorch](https://pytorch.org/)
2. [tqdm](https://github.com/tqdm/tqdm)

A deep learning pipeline to predict  parameters of a circle ( center (x, y) and radius (r)) embedded in a 
noisy/occluded image. 

Trained models and backbone located here: [./results/models](./results/models).

A quick test can be run using the `param_tester.py` script. This script uses the `iou()` defined in [./cv_practical/main.py](./cv_practical/main.py) to measure performance of  the trained model here: [./results/models/cdnet-*.pth](./results/models/cdnet-*.pth) using a dataset of 1000 images as defined in the evaluation criteria.

##### Results

1. Results of the denoising  pipeline using DnCNN

   1. Total number of parameters: 112192

   2. | noisy image input                                            | target image                                                 | denoised output                                              |
      | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
      | ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\images\0.png) | ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\images\0_targ.png) | ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\images\0_pred.png) |
      | ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\images\1.png) | ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\images\1_targ.png) | ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\images\1_pred.png) |

      

   3. Model training and test loss for DnCNN:

      ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\dncnn_plot.png)

2. Results for CDNet (End to end model):

   1. Total number of parameters (excluding the DnCNN backbone): 444467
   2. Model test score (IOU metric):

   ​										 ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\test_score.png)

   3. Model training, test and IOU (train in orange, test in blue and IOU in green): 

   ![](C:\Users\dhruv\Development\git\circle_detection_dl\results\cdnet_plot.png)