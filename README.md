# Assignment – SESSION 6: MNIST Model Experiments

## Objective

The task is to build and train models for the MNIST digit classification dataset under strict constraints:

 - **Target Accuracy**: ≥ 99.4% (must be consistently maintained across the last few epochs, not a single occurrence).
 - **Training Epochs**: ≤ 15 epochs.
 - **Model Parameters**: ≤ 8,000 parameters.

---

## Experiments

### Model1
 - **Target**:
    - Getting Setup Right - Data Loaders, Data Sets, Training, Testing, Training Logs. 
    - Define Model with < 100k parameters with decent training and testing accuracies.
 - **Result**:
    - Got a model with 10790 parameters.
    - Best Training Accuracy under 15 epochs: 99.22%
    - Best Testing Accuracy under 15 epochs: 98.80%
 - **Training Logs**:
<pre style="font-size:10px;">
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

### Model2
 - **Target**:
 - **Result**:
 - **Analysis**:
 - **Training Logs**:
 - **File Link**:

### Model3
 - **Target**:
 - **Result**:
 - **Analysis**:
 - **Training Logs**:
 - **File Link**:

