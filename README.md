# Neural Networks Project - Gesture Recognition
## Abstract
This project aims to develop an advanced hand gesture recognition system utilizing the latest advancements in neural network architectures. The objective is to create a highly accurate and efficient model capable of real-time gesture classification from video input.

## Introduction
Hand gesture recognition is a critical component in human-computer interaction (HCI), enabling intuitive and non-verbal communication with digital devices. Recent advances in deep learning have significantly improved the accuracy and robustness of gesture recognition systems. This project leverages state-of-the-art neural network models, such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformer-based architectures, to achieve superior performance in gesture recognition tasks.

## Methodology

### 1. Data Collection and Preprocessing
- **Dataset Acquisition:** Utilize publicly available datasets such as the American Sign Language (ASL) dataset, Dynamic Hand Gesture dataset, and others. Augment the dataset with synthetic data using techniques like GANs (Generative Adversarial Networks) to increase diversity.
- **Preprocessing:** Implement image normalization, scaling, and augmentation (rotation, flipping, cropping) to enhance the model’s robustness. Extract frames from video sequences and apply optical flow techniques to capture motion information.

### 2. Model Architecture
- **CNN-Based Models:** Implement deep CNNs like VGGNet, ResNet, and InceptionNet for feature extraction from hand images. Fine-tune pre-trained models on the gesture datasets to leverage transfer learning.
- **RNN-Based Models:** Employ Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks to capture temporal dependencies in gesture sequences. Combine CNNs for spatial feature extraction with RNNs for temporal modeling.
- **Transformer-Based Models:** Explore Vision Transformers (ViT) and TimeSformer for capturing both spatial and temporal information in hand gesture recognition tasks. Fine-tune these models on the collected datasets for optimal performance.

### 3. Training and Optimization
- **Loss Function:** Use categorical cross-entropy loss for classification tasks. Implement custom loss functions if necessary to handle class imbalance and other specific challenges.
- **Optimization Techniques:** Utilize advanced optimization algorithms like AdamW, Ranger, and learning rate scheduling strategies (e.g., cyclic learning rates) to enhance convergence speed and model performance.
- **Regularization:** Apply dropout, batch normalization, and data augmentation techniques to prevent overfitting. Use early stopping based on validation loss to halt training when performance plateaus.

### 4. Evaluation and Metrics
- **Performance Metrics:** Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix. Conduct cross-validation to ensure the model's robustness and generalizability.
- **Real-Time Performance:** Assess the model’s inference time and latency to ensure suitability for real-time applications. Optimize the model using techniques like model pruning, quantization, and deployment on edge devices (e.g., TensorRT, ONNX).

### 5. Implementation and Deployment
- **Frameworks and Libraries:** Use deep learning frameworks like TensorFlow, and OpenCV for model development and deployment.
- **User Interface:** Develop a user-friendly interface using tools like Flask or Django for web-based applications, or integrate with hardware using platforms like Raspberry Pi for standalone applications.
- **Real-Time Integration:** Implement real-time gesture recognition in applications such as virtual reality (VR) environments, gaming, and assistive technologies for individuals with disabilities.

## References
Cite pertinent studies, databases, and other materials that helped shape the project's development.

## Developers

Vineet Tiwari [https://www.linkedin.com/in/vineet-tiwari-838228147/]

## Problem Statement
## Gesture Commands
The system will recognize the following gestures, each associated with a specific command:

- **Thumbs up:** Increase the volume
- **Thumbs down:** Decrease the volume
- **Left swipe:** Jump backwards 10 seconds
- **Right swipe:** Jump forward 10 seconds
- **Stop:** Pause the movie

## Methodology
As a data scientist at a home electronics company specializing in state-of-the-art smart televisions, our goal is to enhance the user experience by introducing a gesture-based control feature. This feature will allow users to control the TV using five distinct hand gestures captured by a webcam mounted on the TV. Each gesture will correspond to a specific command, eliminating the need for a remote control.
### Data Collection and Preprocessing
1. **Dataset Acquisition**
   - Collect video data of individuals performing each of the five gestures. Each video will consist of 30 frames.
   - Augment the dataset to include variations in lighting, background, and hand orientations to improve model robustness.

2. **Preprocessing**
   - Normalize and scale the images to ensure consistent input size.
   - Apply data augmentation techniques such as rotation, flipping, and cropping.
   - Extract frames from the videos and preprocess them using optical flow techniques to capture motion dynamics.

### Model Architecture
1. **CNN-Based Feature Extraction**
   - Utilize deep Convolutional Neural Networks (CNNs) such as VGGNet, ResNet, and InceptionNet to extract spatial features from each frame.
   - Fine-tune pre-trained CNN models on the gesture dataset to leverage transfer learning.

2. **RNN-Based Temporal Modeling**
   - Implement Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks to capture temporal dependencies across the sequence of frames.
   - Combine CNNs for spatial feature extraction with RNNs for temporal sequence modeling.

3. **Transformer-Based Approach**
   - Explore Vision Transformers (ViT) and TimeSformer architectures to capture both spatial and temporal information simultaneously.
   - Fine-tune these models on the gesture dataset for optimal performance.

### Training and Optimization
1. **Loss Function**
   - Use categorical cross-entropy loss for multi-class gesture classification.

2. **Optimization Techniques**
   - Employ advanced optimizers like AdamW and Ranger.
   - Implement learning rate scheduling strategies, such as cyclic learning rates, to enhance model convergence.

3. **Regularization**
   - Apply dropout and batch normalization to prevent overfitting.
   - Use early stopping based on validation loss to halt training when the model performance plateaus.

### Evaluation and Metrics
1. **Performance Metrics**
   - Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.
   - Conduct cross-validation to ensure model robustness and generalizability.

2. **Real-Time Performance**
   - Assess inference time and latency to ensure the system operates in real-time.
   - Optimize the model using techniques like model pruning and quantization for deployment on edge devices.

### Implementation and Deployment
1. **Frameworks and Libraries**
   - Utilize TensorFlow, PyTorch, and OpenCV for model development and deployment.

2. **User Interface**
   - Develop a user-friendly interface using Flask or Django for web-based applications.
   - Integrate the model with the smart TV platform for seamless user interaction.

3. **Real-Time Integration**
   - Implement the gesture recognition feature to control TV functions like volume adjustment, video playback, and pausing movies.

## Results
- Quantitative evaluation of model performance on benchmark datasets.
- Qualitative examples demonstrating successful gesture recognition in various scenarios.
- Comparison of different model architectures to justify the selection of the final model.


## Understanding the Dataset
A few hundred videos that have been categorized into one of the five classes make up the training data. Every video, which lasts for two to three seconds on average, is composed of thirty frames, or images. These films were captured by different individuals using a webcam—the same kind that the smart TV would use—while making one of the five gestures. 

A [zip](https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL) file contains the data. A 'train' and a 'val' folder with two CSV files each can be found in the zip file.

## Model Overview

| Model Name                  | Model Type               | Number of parameters | More Augment Data | Highest Validation accuracy | Corresponding Training accuracy | Observations                                             | Data Science Observations                                           |
|-----------------------------|--------------------------|----------------------|-------------------|-----------------------------|---------------------------------|----------------------------------------------------------|----------------------------------------------------------------------|
| ModelConv3D3 - conv_3d1     | Conv3D                   | 1117061              | No                | 0.25                        | 0.95                            | The model fits too closely. Use cropping to ensure better results and generalize the model. | Overfitting is evident with high training accuracy but low validation accuracy. Regularization techniques like dropout could help. |
| ModelConv3D3 - conv_3d2     | Conv3D                   | 3638981              | No                | 0.75                        | 0.90                            | It's not an over-fitting model. We'll attempt further improvement by adding regularization and dropout layers. | The model performs better and shows potential. Fine-tuning hyperparameters might further improve results. |
| ModelConv3D3 - conv_3d3     | Conv3D                   | 1762613              | No                | 0.20                        | 0.70                            | Due to overfitting and no changes in loss function, it's recommended to add more augment data. | Severe overfitting indicated by a large gap between training and validation accuracies. Data augmentation and early stopping could help. |
| ModelConv3D4 - conv_3d4     | Conv3D                   | 2556533              | No                | 0.20                        | 0.75                            | Due to overfitting and no changes in loss function, it's recommended to add more augment data. | Similar issues as conv_3d3. Consider simplifying the model to avoid overfitting. |
| ModelConv3D5 - conv_3d5     | Conv3D                   | 2556533              | No                | 0.35                        | 0.80                            | Model starts to perform better but it is stopped early, consider using learning rate scheduler. | Improvement seen, but early stopping might have halted learning prematurely. |
| ModelConv3D6 - conv_3d6     | Conv3D                   | 696645               | No                | 0.30                        | 0.85                            | Model starts to perform better but it is stopped early, consider using learning rate scheduler. | Smaller model size might be beneficial. Learning rate adjustments could help optimize performance. |
| ModelConv3D7 - conv_3d7     | Conv3D                   | 504709               | No                | 0.70                        | 0.75                            | Model is performing good                                    | Good balance between training and validation performance. Additional training epochs might improve validation accuracy. |
| RNNCNN1 - rnn_cnn1          | CNN-LSTM                 | 1657445              | No                | 0.84                        | 0.92                            | One of the best results                                    | Strong model performance suggests effective learning. Consider cross-validation to confirm results. |
| ModelConv3D9  - conv_3d9    | Conv3D                   | 3638981              | Yes               | 0.77                        | 0.74                            |                                                          | Data augmentation appears beneficial here. Continue experimenting with augmentations. |
| ModelConv3D10 - conv_3d10   | Conv3D                   | 1762613              | Yes               | 0.45                        | 0.69                            |                                                          | Slight improvement with data augmentation, but still overfitting. Further regularization needed. |
| RNNCNN2 - rnn_cnn2          | CNN LSTM with GRU        | 2573925              | Yes               | 0.82                        | 0.94                            |                                                          | Promising results with CNN LSTM and GRU combination. Ensure robustness with more data. |
| RNNCNN_TL - rnn_cnn_tl      | ImageNet with an LSTM    | 3840453              | NaN               | 0.80                        | 0.98                            |                                                          | Transfer learning shows high performance. Consider reducing the model's complexity if overfitting is observed. |


## Model Selection and Finalization
After extensive experimentation, we finalized **Model 8 - CNN+LSTM** due to its superior performance and efficiency. The selection was based on the following factors:

- **Training Accuracy:** 93%
- **Validation Accuracy:** 85%
- **Number of Parameters:** 1,657,445 (significantly less compared to other models, ensuring faster computation and reduced resource usage)
- **Learning Rate Adjustment:** The learning rate was gradually decreased after 16 epochs, aiding in better convergence and reducing overfitting.

## Results
- Quantitative evaluation of model performance on benchmark datasets.
- Qualitative examples demonstrating successful gesture recognition in various scenarios.
- Comparison of different model architectures to justify the selection of the final model.