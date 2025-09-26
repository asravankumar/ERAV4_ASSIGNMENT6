"""
 - Target:
    - Address Overfitting, late convergence with lesser number of parameters.
 - Result:
    - Was able to achieve > 99% testing accuracy on all epochs greater than 10 with less than 8k parameters. But falls short of the required 99.4% accuracy.
    - Best Training Accuracy under 15 epochs: 99.01%
    - Best Testing Accuracy under 15 epochs: 99.32%
 - Analysis:
    - Use of Batch Normalization has converged both the training and testing accuracies in lesser number of epochs there by drastically helping in converging faster for the required accuracies in less than 15 epochs.
    - As the model started overfitting after few epochs in previous iteration, using a dropout of 0.05 on all layers contributed immensely. It provided the necessary regularization there by slightly reducing the training accuracies.
    - By replacing the last convolution layer with GAP has drastically reduced the overall number of parameters and at the same time preserving the overall model performance.
    - But this model falls short of required testing accuracies. Perhaps, we need to play by adding augmentation to improve the testing accuracies.
    - At the same time, we also need to tune the model with different Learning Rates.
"""

import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.05

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26, Receptive Field - 3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 24, Receptive Field - 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, Receptive Field - 6
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 12, Receptive Field - 6

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 10, Receptive Field - 10

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 8, Receptive Field - 14

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output size = 6, Receptive Field - 18

        # OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 6, Receptive Field - 18
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, Receptive Field - 28


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.pool1(x)
        x = self.convblock4(x)

        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)