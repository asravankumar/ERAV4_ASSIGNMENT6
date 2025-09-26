# Assignment – SESSION 6: MNIST Model Experiments

## Objective

The task is to build and train models for the MNIST digit classification dataset under strict constraints:

 - **Target Accuracy**: ≥ 99.4% (must be consistently maintained across the last few epochs, not a single occurrence).
 - **Training Epochs**: ≤ 15 epochs.
 - **Model Parameters**: ≤ 8,000 parameters.


---
## Experiments
---
### Model1
 - **Target**:
    - Getting Setup Right - Data Loaders, Data Sets, Training, Testing, Training Logs. 
    - Define Model with < 100k parameters with decent training and testing accuracies.
 - **Result**:
    - Got a model with 10790 parameters.
    - Best Training Accuracy under 15 epochs: 99.22%
    - Best Testing Accuracy under 15 epochs: 98.80%
 - **Training Logs**:
<pre style="font-size:4px;">
EPOCH: 0
Loss=0.3600336015224457 Batch_id=468 Accuracy=33.99: 100%|██████████| 469/469 [00:13<00:00, 34.37it/s]
Test set: Average loss: 0.3240, Accuracy: 9017/10000 (90.17%)

EPOCH: 1
Loss=0.026120617985725403 Batch_id=468 Accuracy=94.45: 100%|██████████| 469/469 [00:13<00:00, 34.67it/s]
Test set: Average loss: 0.1066, Accuracy: 9676/10000 (96.76%)

EPOCH: 2
Loss=0.0994456335902214 Batch_id=468 Accuracy=97.01: 100%|██████████| 469/469 [00:13<00:00, 34.09it/s]
Test set: Average loss: 0.0789, Accuracy: 9736/10000 (97.36%)

EPOCH: 3
Loss=0.13692565262317657 Batch_id=468 Accuracy=97.76: 100%|██████████| 469/469 [00:13<00:00, 34.59it/s]
Test set: Average loss: 0.0677, Accuracy: 9774/10000 (97.74%)

EPOCH: 4
Loss=0.07584532350301743 Batch_id=468 Accuracy=98.08: 100%|██████████| 469/469 [00:14<00:00, 33.28it/s]
Test set: Average loss: 0.0532, Accuracy: 9817/10000 (98.17%)

EPOCH: 5
Loss=0.03837588056921959 Batch_id=468 Accuracy=98.38: 100%|██████████| 469/469 [00:13<00:00, 34.32it/s]
Test set: Average loss: 0.0492, Accuracy: 9848/10000 (98.48%)

EPOCH: 6
Loss=0.04064289480447769 Batch_id=468 Accuracy=98.52: 100%|██████████| 469/469 [00:13<00:00, 34.09it/s]
Test set: Average loss: 0.0575, Accuracy: 9820/10000 (98.20%)

EPOCH: 7
Loss=0.013719498179852962 Batch_id=468 Accuracy=98.75: 100%|██████████| 469/469 [00:13<00:00, 34.52it/s]
Test set: Average loss: 0.0512, Accuracy: 9847/10000 (98.47%)

EPOCH: 8
Loss=0.020614534616470337 Batch_id=468 Accuracy=98.79: 100%|██████████| 469/469 [00:13<00:00, 33.80it/s]
Test set: Average loss: 0.0477, Accuracy: 9844/10000 (98.44%)

EPOCH: 9
Loss=0.08759859949350357 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:13<00:00, 34.25it/s]
Test set: Average loss: 0.0484, Accuracy: 9859/10000 (98.59%)

EPOCH: 10
Loss=0.007742138113826513 Batch_id=468 Accuracy=98.90: 100%|██████████| 469/469 [00:13<00:00, 34.51it/s]
Test set: Average loss: 0.0450, Accuracy: 9865/10000 (98.65%)

EPOCH: 11
Loss=0.05451782047748566 Batch_id=468 Accuracy=99.02: 100%|██████████| 469/469 [00:13<00:00, 34.15it/s]
Test set: Average loss: 0.0448, Accuracy: 9854/10000 (98.54%)

EPOCH: 12
Loss=0.01963711716234684 Batch_id=468 Accuracy=99.10: 100%|██████████| 469/469 [00:13<00:00, 34.01it/s]
Test set: Average loss: 0.0416, Accuracy: 9868/10000 (98.68%)

EPOCH: 13
Loss=0.06757549941539764 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:14<00:00, 32.77it/s]
Test set: Average loss: 0.0443, Accuracy: 9870/10000 (98.70%)

EPOCH: 14
Loss=0.026263413950800896 Batch_id=468 Accuracy=99.22: 100%|██████████| 469/469 [00:13<00:00, 34.48it/s]
Test set: Average loss: 0.0413, Accuracy: 9880/10000 (98.80%)
</pre>
 - **Analysis**:
    - Even though the model achives 98.80% testing accuracy by epoch 14, but it requires a high number of epochs. Adding **Batch Normalization** would stabilize the training, allowing to hit ~99% much faster.
    - Inefficient use of final layers. Use of **Global Average Pooling** would definitely reduce the number of parameters drastically.
    - Post 5-6 epochs, the model starts over-fitting. We definitely need to add regularization to address this.
 - **File Link**:
    - [Model1 File Link](model1.py)
---
### Model2
 - **Target**:
    - Address Overfitting, late convergence with lesser number of parameters.
 - **Result**:
    - Was able to achieve > 99% testing accuracy on all epochs greater than 10 with less than 8k parameters. But falls short of the required 99.4% accuracy.
    - Best Training Accuracy under 15 epochs: 99.01%
    - Best Testing Accuracy under 15 epochs: 99.32%
 - **Training Logs**:
<pre style="font-size:4px;">
EPOCH: 0
Loss=0.1843942403793335 Batch_id=468 Accuracy=77.96: 100%|██████████| 469/469 [00:13<00:00, 35.48it/s]
Test set: Average loss: 0.1667, Accuracy: 9637/10000 (96.37%)

EPOCH: 1
Loss=0.06241855025291443 Batch_id=468 Accuracy=97.09: 100%|██████████| 469/469 [00:13<00:00, 34.74it/s]
Test set: Average loss: 0.0713, Accuracy: 9819/10000 (98.19%)

EPOCH: 2
Loss=0.023404819890856743 Batch_id=468 Accuracy=97.79: 100%|██████████| 469/469 [00:14<00:00, 32.84it/s]
Test set: Average loss: 0.0568, Accuracy: 9852/10000 (98.52%)

EPOCH: 3
Loss=0.05270249769091606 Batch_id=468 Accuracy=98.15: 100%|██████████| 469/469 [00:13<00:00, 35.80it/s]
Test set: Average loss: 0.0440, Accuracy: 9879/10000 (98.79%)

EPOCH: 4
Loss=0.03701097145676613 Batch_id=468 Accuracy=98.36: 100%|██████████| 469/469 [00:13<00:00, 35.71it/s]
Test set: Average loss: 0.0420, Accuracy: 9878/10000 (98.78%)

EPOCH: 5
Loss=0.027281438931822777 Batch_id=468 Accuracy=98.58: 100%|██████████| 469/469 [00:13<00:00, 35.39it/s]
Test set: Average loss: 0.0399, Accuracy: 9889/10000 (98.89%)

EPOCH: 6
Loss=0.02308700419962406 Batch_id=468 Accuracy=98.58: 100%|██████████| 469/469 [00:12<00:00, 36.57it/s]
Test set: Average loss: 0.0333, Accuracy: 9902/10000 (99.02%)

EPOCH: 7
Loss=0.04752656817436218 Batch_id=468 Accuracy=98.67: 100%|██████████| 469/469 [00:13<00:00, 35.57it/s]
Test set: Average loss: 0.0328, Accuracy: 9893/10000 (98.93%)

EPOCH: 8
Loss=0.011922412551939487 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:13<00:00, 34.48it/s]
Test set: Average loss: 0.0304, Accuracy: 9911/10000 (99.11%)

EPOCH: 9
Loss=0.05148118734359741 Batch_id=468 Accuracy=98.80: 100%|██████████| 469/469 [00:13<00:00, 35.56it/s]
Test set: Average loss: 0.0345, Accuracy: 9897/10000 (98.97%)

EPOCH: 10
Loss=0.0063825552351772785 Batch_id=468 Accuracy=98.80: 100%|██████████| 469/469 [00:12<00:00, 36.29it/s]
Test set: Average loss: 0.0267, Accuracy: 9932/10000 (99.32%)

EPOCH: 11
Loss=0.021838247776031494 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:13<00:00, 35.56it/s]
Test set: Average loss: 0.0288, Accuracy: 9909/10000 (99.09%)

EPOCH: 12
Loss=0.03239915519952774 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:12<00:00, 36.33it/s]
Test set: Average loss: 0.0287, Accuracy: 9908/10000 (99.08%)

EPOCH: 13
Loss=0.017107730731368065 Batch_id=468 Accuracy=98.99: 100%|██████████| 469/469 [00:13<00:00, 35.34it/s]
Test set: Average loss: 0.0268, Accuracy: 9917/10000 (99.17%)

EPOCH: 14
Loss=0.016882238909602165 Batch_id=468 Accuracy=99.01: 100%|██████████| 469/469 [00:13<00:00, 35.49it/s]
Test set: Average loss: 0.0255, Accuracy: 9910/10000 (99.10%)
</pre>

 - **Analysis**:
    - Use of Batch Normalization has converged both the training and testing accuracies in lesser number of epochs there by drastically helping in converging faster for the required accuracies in less than 15 epochs.
    - As the model started overfitting after few epochs in previous iteration, using a dropout of 0.05 on all layers contributed immensely. It provided the necessary regularization there by slightly reducing the training accuracies.
    - By replacing the last convolution layer with GAP has drastically reduced the overall number of parameters and at the same time preserving the overall model performance.
    - But this model falls short of required testing accuracies. Perhaps, we need to play by adding augmentation to improve the testing accuracies.
    - At the same time, we also need to tune the model with different Learning Rates.
 - **File Link**:
    - [Model2 File Link](model2.py)
---
### Model3
 - **Target**:
    - Achieve at least 99.40% testing accuracy consistently for few epochs under 8k parameters.
 - **Result**:
    - Able to achieve the desired output with **6136** parameters.
 - **Training Logs**:
<pre style="font-size:4px;">
EPOCH: 0
Loss=0.12491422891616821 Batch_id=468 Accuracy=86.49: 100%|██████████| 469/469 [00:24<00:00, 19.35it/s]
Test set: Average loss: 0.0656, Accuracy: 9808/10000 (98.08%)

EPOCH: 1
Loss=0.026768239215016365 Batch_id=468 Accuracy=96.90: 100%|██████████| 469/469 [00:24<00:00, 18.81it/s]
Test set: Average loss: 0.0450, Accuracy: 9859/10000 (98.59%)

EPOCH: 2
Loss=0.15286700427532196 Batch_id=468 Accuracy=97.59: 100%|██████████| 469/469 [00:26<00:00, 17.71it/s]
Test set: Average loss: 0.0308, Accuracy: 9896/10000 (98.96%)

EPOCH: 3
Loss=0.029769212007522583 Batch_id=468 Accuracy=97.77: 100%|██████████| 469/469 [00:25<00:00, 18.67it/s]
Test set: Average loss: 0.0323, Accuracy: 9896/10000 (98.96%)

EPOCH: 4
Loss=0.022351177409291267 Batch_id=468 Accuracy=97.86: 100%|██████████| 469/469 [00:24<00:00, 19.03it/s]
Test set: Average loss: 0.0269, Accuracy: 9923/10000 (99.23%)

EPOCH: 5
Loss=0.04780856892466545 Batch_id=468 Accuracy=98.08: 100%|██████████| 469/469 [00:23<00:00, 19.55it/s]
Test set: Average loss: 0.0278, Accuracy: 9924/10000 (99.24%)

EPOCH: 6
Loss=0.04058980941772461 Batch_id=468 Accuracy=98.38: 100%|██████████| 469/469 [00:24<00:00, 18.94it/s]
Test set: Average loss: 0.0210, Accuracy: 9937/10000 (99.37%)

EPOCH: 7
Loss=0.01003444753587246 Batch_id=468 Accuracy=98.43: 100%|██████████| 469/469 [00:24<00:00, 19.21it/s]
Test set: Average loss: 0.0207, Accuracy: 9937/10000 (99.37%)

EPOCH: 8
Loss=0.020389266312122345 Batch_id=468 Accuracy=98.53: 100%|██████████| 469/469 [00:24<00:00, 19.01it/s]
Test set: Average loss: 0.0197, Accuracy: 9941/10000 (99.41%)

EPOCH: 9
Loss=0.08403483033180237 Batch_id=468 Accuracy=98.55: 100%|██████████| 469/469 [00:24<00:00, 19.05it/s]
Test set: Average loss: 0.0188, Accuracy: 9943/10000 (99.43%)

EPOCH: 10
Loss=0.013831782154738903 Batch_id=468 Accuracy=98.56: 100%|██████████| 469/469 [00:24<00:00, 18.87it/s]
Test set: Average loss: 0.0188, Accuracy: 9942/10000 (99.42%)

EPOCH: 11
Loss=0.0744112879037857 Batch_id=468 Accuracy=98.54: 100%|██████████| 469/469 [00:24<00:00, 19.28it/s]
Test set: Average loss: 0.0189, Accuracy: 9940/10000 (99.40%)

EPOCH: 12
Loss=0.10935012251138687 Batch_id=468 Accuracy=98.56: 100%|██████████| 469/469 [00:25<00:00, 18.69it/s]
Test set: Average loss: 0.0185, Accuracy: 9942/10000 (99.42%)

EPOCH: 13
Loss=0.12498316913843155 Batch_id=468 Accuracy=98.56: 100%|██████████| 469/469 [00:24<00:00, 18.95it/s]
Test set: Average loss: 0.0184, Accuracy: 9940/10000 (99.40%)

EPOCH: 14
Loss=0.03961181640625 Batch_id=468 Accuracy=98.60: 100%|██████████| 469/469 [00:24<00:00, 18.99it/s]
Test set: Average loss: 0.0190, Accuracy: 9940/10000 (99.40%)
</pre>
 - **Analysis**:
    - Adding Image augmentation techniques to the previous model with slight changes in the network has definitely improved the testing accuracy.
    - Had played with different learning rates for SGD and finally with StepLR and step_size of 6, was able to achieve the desired result.
    - Was able to achieve >= 99.40% testing accuracy consistently from epoch 9.
 - **File Link**:
    - [Model3 File Link](model3.py)
---

## Conclusion

Step-by-step iterative experimentation was performed to achieve the desired model.
 - Starting from skeleton to a basic minimal model with ~10k parameters.
 - Then by batch normalization we could converge faster and by adding regularization we address overfitting. But the testing accuracies needed to improve.
 - By adding image augmentation techniques and playing with the learning rates, we could achieve the following best model with desired output.
    - 6,136  parameters.
    - \>= 99.40% testing accuracies from 9th epoch.

### Best Model Architecture

Summary of the model

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
       BatchNorm2d-2            [-1, 8, 26, 26]              16
              ReLU-3            [-1, 8, 26, 26]               0
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
       BatchNorm2d-6           [-1, 16, 24, 24]              32
              ReLU-7           [-1, 16, 24, 24]               0
           Dropout-8           [-1, 16, 24, 24]               0
         MaxPool2d-9           [-1, 16, 12, 12]               0
           Conv2d-10            [-1, 8, 12, 12]             128
      BatchNorm2d-11            [-1, 8, 12, 12]              16
             ReLU-12            [-1, 8, 12, 12]               0
           Conv2d-13            [-1, 8, 10, 10]             576
      BatchNorm2d-14            [-1, 8, 10, 10]              16
             ReLU-15            [-1, 8, 10, 10]               0
          Dropout-16            [-1, 8, 10, 10]               0
           Conv2d-17             [-1, 12, 8, 8]             864
      BatchNorm2d-18             [-1, 12, 8, 8]              24
             ReLU-19             [-1, 12, 8, 8]               0
          Dropout-20             [-1, 12, 8, 8]               0
           Conv2d-21             [-1, 12, 6, 6]           1,296
      BatchNorm2d-22             [-1, 12, 6, 6]              24
             ReLU-23             [-1, 12, 6, 6]               0
          Dropout-24             [-1, 12, 6, 6]               0
           Conv2d-25             [-1, 16, 4, 4]           1,728
      BatchNorm2d-26             [-1, 16, 4, 4]              32
             ReLU-27             [-1, 16, 4, 4]               0
          Dropout-28             [-1, 16, 4, 4]               0
        AvgPool2d-29             [-1, 16, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             160
================================================================
Total params: 6,136
Trainable params: 6,136
Non-trainable params: 0
----------------------------------------------------------------
```
