## Overview

hw4p2 is about building Automatic Speech Recognition using Transformer based architecture.
View run Angela Yang_fbank_Transformer_ENC-4-8_DEC-4-8_512_1500_AdamW_CosineAnnealing_token_char at: https://wandb.ai/anqiyang00-carnegie-mellon-university/HW4P2-Fall/runs/qo02vntb


## How to Run

Simply run through the whole juypter notebook

---

## Dataset Configuration
- **`root`**: Path to the root directory of the dataset. *(Example: `/785/hw4p2/hw4p2`)*
- **`unpaired_text_partition`**: Name of unpaired text data for language model pre-training. *(Default: `text-for-LM`)*
- **`train_partition`**: Name of the training data partition. *(Default: `train-clean-100`)*
- **`val_partition`**: Name of the validation data partition. *(Default: `dev-clean`)*
- **`test_partition`**: Name of the test data partition. *(Default: `test-clean`)*
- **`NUM_WORKERS`**: Number of workers for data loading. *(Default: 4)* tried 8,10,12 but 12 may cause GPU out of memory issue
- **`subset`**: Proportion of the dataset to load for debugging/testing. *(Default: 1.0)*
- **`token_type`**: Type of tokenization. Options: `char`, `1k`, `10k`. Tired char
- **`feat_type`**: Feature type for extraction. Options: `fbank`, `mfcc`. Tried fbank 
- **`num_feats`**: Number of features for extraction. *(Default: 80 for `fbank`, 12-20 for `mfcc`)* 
- **`batch_size`**: Batch size for training and evaluation. *(Default: 32)* Tried 8, 16, 64,128. 64 and 128 may lead to GPU out of memory issue 
- **`norm`**: Normalization type. Options:
  - `global_mvn`: Global mean and variance normalization.
  - `cepstral`: Cepstral normalization.
tried cepstral
---

## SpecAugment Configuration
- **`specaug`**: Enable SpecAugment. *(Default: True)*
- **`specaug_conf`**: Parameters for SpecAugment:
  - **`apply_freq_mask`**: Apply frequency masking. *(Default: True)*
  - **`freq_mask_width_range`**: Frequency mask width range. *(Default: 4)*
  - **`num_freq_mask`**: Number of frequency masks. *(Default: 4)*
  - **`apply_time_mask`**: Apply time masking. *(Default: True)*
  - **`time_mask_width_range`**: Time mask width range. *(Default: 80)* Change to 15 improved model performance
  - **`num_time_mask`**: Number of time masks. *(Default: 8)*

---

## Network Specifications
- **`d_model`**: Model dimension for transformers or similar architectures. *(Default: 256)* Tired 512
- **`d_ff`**: Feed-forward network dimension. *(Default: 1024)* Tried 1500, 1536

---

## Embedding Specifications
- **`time_stride`**: Time-wise downsampling factor. *(Default: 4)* tried 2
- **`feature_stride`**: Feature-wise downsampling factor. *(Default: 4)* tried 2 and 1 
- **`embed_dropout`**: Dropout rate for embedding layers. *(Default: 0.2)*

---

## Encoder Specifications
- **`enc_dropout`**: Dropout rate for encoder layers. *(Default: 0.2)*
- **`enc_num_layers`**: Number of layers in the encoder. *(Default: 2)* tried 4,5,8,12. 8 and 12 lead to GPU out of memory 
- **`enc_num_heads`**: Number of attention heads in the encoder. *(Default: 2)* tired 8 

---

## Decoder Specifications
- **`dec_dropout`**: Dropout rate for decoder layers. *(Default: 0.2)*
- **`dec_num_layers`**: Number of layers in the decoder. *(Default: 2)* Tried 2,5,8
- **`dec_num_heads`**: Number of attention heads in the decoder. *(Default: 2)* Tried 4,8

---

## Training Parameters
- **`use_wandb`**: Enable Weights & Biases integration. *(Default: True)*
- **`use_ctc`**: Use Connectionist Temporal Classification (CTC) loss. *(Default: True)*
- **`ctc_weight`**: Weight for CTC loss in multi-task training. *(Default: 0.4)*
- **`optimizer`**: Optimizer type. Options: `Adam`, `AdamW`, `SGD`.
- **`momentum`**: Momentum value for optimizers (if applicable). *(Default: 0.1)*
- **`nesterov`**: Enable Nesterov momentum. *(Default: True)*
- **`learning_rate`**: Initial learning rate for training. *(Default: 2e-4)*
- **`scheduler`**: Learning rate scheduler. Options: `ReduceLR`, `CosineAnnealing`.
- **`factor`**: Factor for `ReduceLR` scheduler. *(Default: 0.2)*
- **`patience`**: Patience for `ReduceLR` scheduler. *(Default: 2)*
- **`epochs`**: Total number of training epochs. *(Default: 80)*

---
Also tried the pretrain LM decoder for 20 epoches, but going to full train directly works better 