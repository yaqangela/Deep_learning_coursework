
# Image Verification using CNNs

This repository contains the implementation for Homework 2 of the **11785: Introduction to Deep Learning** course. The project addresses the problem of **Image Verification**, where the aim is to identify whether pairs of face images match.

## Project Overview

The main objectives of this homework are:
1. Train a Convolutional Neural Network (CNN) for **classification** over 8,631 different identities.
2. Extract **face embeddings** from trained models and use them for the verification task, identifying whether two face images belong to the same identity.

## Dataset

- **Number of Classes:** 8,631
- **Number of Training Images:** 431,549
- **Image Shape:** `[3, 224, 224]`

## Model Training

### Initial Training Configuration

- **Batch Size:** 256
- **Learning Rate:** 1e-3
- **Epochs:** 80 (20 recommended for early submission)
- **Weight Decay:** 0.05
- **Dropout:** 0.2
- **Layer Scale Initialization:** 1e-6
- **Training Batches:** 1,686
- **Validation Batches:** 169
- **Model architecure:** Restnet-50 with cutmix

The initial model training focuses on classification over 8,631 identities. The trained model is then fine-tuned using ArcFace loss for better face verification. However, I reached the high-cutoff by using Restnet-50 and cutmix, so I did not train with arcface loss fine tune. 
I tried Convnext- tiny, restnet-34 with bottleneck block, and Restnet-50 with basic block. Convnext- tiny reached 19 EER at around 70 epochs and stop decreasing. 

### Fine-Tuning Configuration

- **Learning Rate:** 5e-4
- **Momentum:** 0.9
- **Weight Decay:** 1e-4
- **Scheduler:** `ReduceLROnPlateau`
  - **Patience:** 3
  - **Threshold Mode:** 'rel'
  - **Factor:** 0.7
  - **Verbose:** True
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 20
- **Batch Size:** 256

## Dependencies

Ensure that the following libraries are installed before running the code:

- PyTorch
- torchvision
- numpy
- tqdm

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```




## Results and Evaluation

Once trained, the model extracts face embeddings from pairs of images and uses a similarity metric to determine if the pairs match or not. Evaluation metrics such as accuracy, precision, recall, and F1-score are used to assess performance.

## Notes

- Adjust the `data_dir`, `data_ver_dir`, and `checkpoint_dir` fields in the configuration dictionaries as needed.
- This project is computationally intensive; using a GPU is highly recommended for training and fine-tuning.

## License

This project is part of coursework for **11785: Introduction to Deep Learning** at Carnegie Mellon University. Please use the code responsibly and adhere to the academic integrity policy.
