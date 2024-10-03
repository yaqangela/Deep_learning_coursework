# Project Title

This project involves experimenting with various neural network architectures to achieve optimal performance. Below are the running instructions, architectures tried, and best configurations.

## Running Instructions
To run this project, simply execute the notebook:


## Configuration Overview

Below is the updated configuration used for this project:

- **Number of Epochs:** 40
- **Batch Size:** 8192
- **Context Size:** 25
- **Initial Learning Rate:** 0.003 (tried 0.005,0.001,0.002)

### Model Parameters:
- **Architecture:** MLP (Multi-Layer Perceptron)
- **Dropout Rate:** 0.2 (tried 0.5,0.4,0.3)
- **Time Mask:** 10
- **Frequency Mask:** 5

### Scheduler Parameters:
- **Scheduler Type:** Cosine Annealing
  - **T_max:** 10 (Maximum number of epochs for cosine annealing)
  - **Minimum Learning Rate (eta_min):** 1e-5

### Optimizer Parameters:
- **Optimizer Type:** Adam
- **Weight Decay:** 1e-4 (Regularization parameter)

### Weight Initialization:
- **Method:** Kaiming Uniform
- **Nonlinearity:** GELU (Used for activation during initialization)

## Best Combination

Based on experiments, the following combination yielded the best performance:

- **Architecture:** 8 layers of Diamond with GELU activation and batch normalization
- **Dropout Rate:** 0.2
- **Learning Rate Scheduler:** Cosine Annealing
- **Weight Initialization:** Kaiming
- **Context Size:** 25
- **Batch Size:** 8192


### Network Configuration
Here is the best-performing network architecture:

Network(
  (model): Sequential(
    (0): Linear(in_features=1428, out_features=512, bias=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): GELU(approximate='none')
    (3): Linear(in_features=512, out_features=1024, bias=True)
    (4): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): GELU(approximate='none')
    (6): Dropout(p=0.2, inplace=False)
    (7): Linear(in_features=1024, out_features=2048, bias=True)
    (8): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): GELU(approximate='none')
    (10): Dropout(p=0.2, inplace=False)
    (11): Linear(in_features=2048, out_features=3000, bias=True)
    (12): BatchNorm1d(3000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): GELU(approximate='none')
    (14): Dropout(p=0.2, inplace=False)
    (15): Linear(in_features=3000, out_features=2048, bias=True)
    (16): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (17): GELU(approximate='none')
    (18): Dropout(p=0.2, inplace=False)
    (19): Linear(in_features=2048, out_features=1024, bias=True)
    (20): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (21): GELU(approximate='none')
    (22): Dropout(p=0.2, inplace=False)
    (23): Linear(in_features=1024, out_features=512, bias=True)
    (24): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (25): GELU(approximate='none')
    (26): Dropout(p=0.2, inplace=False)
    (27): Linear(in_features=512, out_features=42, bias=True)
  )
)
