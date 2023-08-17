# Discriminating between Chihuahuas and Muffins: A Deep Learning Approach

### Abstract
Image recognition and classification have posed significant technological challenges. However, the advent of deep learning architectures like Convolutional Neural Networks (CNNs) has showcased their ability to achieve remarkable accuracy in such endeavors. This project aims to exhibit how transfer learning methods, employing feature extraction and data augmentation, effectively address these intricate problems. This becomes particularly crucial in scenarios where complexity surges, especially when discerning between highly similar images that pertain to entirely distinct contexts.

### Introduction

Throughout the 2010s, the field of deep learning made remarkable strides, particularly in addressing anticipated tasks like object classification, speech recognition, text analysis, image synthesis, and more. Notably, AI competitions such as the "ImageNet challenge" played a pivotal role in catapulting convolutional architectures into the limelight. These architectures gained immense popularity for their effectiveness in resolving object recognition and classification dilemmas.

In the field of AI and computer vision, we address the challenge of distinguishing between objects that share close similarities, such as Chihuahuas and muffins.

![1*bt-E2YcPafjiPbZFDMMmNQ](https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/4c333fc3-e555-4c2e-9327-04492e69f1d3)

The Chihuahua vs. Muffin Challenge in the field of AI/ML introduces a captivating test of image classification algorithms. This unique challenge centers on differentiating between images of Chihuahua dogs and muffins, two subjects that can surprisingly share visual similarities.


### Methods


#### Data Augmetation
'ImageDataGenerator' class from the Keras library to perform data augmentation. The 'ImageDataGenerator' class helps to generate batches of augmented images seamlessly during training. There are two generators, train_data and test_data, which will load batches of augmented and unmodified images, respectively, from the specified directories. The images are being resized to a target size of 150x150 pixels, and the class_mode is set to 'binary', indicating binary classification problem.

**Rotation Range:** Images are rotated by an angle of 40. This helps the model become invariant to rotation.

**Width and Height Shift Range:** The images are shifted horizontally and vertically by a proportions of 0.2. This simulates changes in perspective and object positioning.

**Zoom Range:** Images are zoomed in and out by a factors of 0.2. This helps the model learn to recognize objects at different scales.

**Horizontal Flip:** Images are flipped horizontally with a certain probability. This helps the model become invariant to horizontal flips.

**Rescale:** The pixel values of the images are rescaled to a range of [0, 1] by dividing them by 255. This standardizes the pixel values.

#### Simple CNN

#### VGG19
![VGG19Arc](https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/7f6817a0-8151-491e-8b34-2ef688914881)

* Excluding final fully connected layers (classifier) of the VGG19 by setting **include_top=False**. This allows us to customize the classifier for our specific problem.
* Using pre-trained weights from the **ImageNet** dataset. This provides a good starting point for feature extraction.
* Expected input image dimensions: 150x150 pixels with 3 color channels (RGB).
* Classes are set to 2 which specifies the number of output classes as 2 for the binary classification problem.
* **loss='binary_crossentropy':** For binary classification problem (two classes), binary_crossentropy is a suitable loss function. It measures the difference between the predicted probabilities and the true labels for each example in the dataset.
* **optimizer='adam':** 'adam' is a popular optimization algorithm that adapts the learning rate during training based on the statistics of the gradients. It is often used as a good default choice for various types of neural networks.

<br>

#### DenseNet121

![DenseNet drawio](https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/40c5abc5-e4f1-4c7d-8e64-d0ad6abcf3c8)

Construction of a binary image classification model using transfer learning with a pre-trained DenseNet-121 model as the feature extractor.

**Dense block** is composed of multiple densely connected layers. Each layer within a dense block receives input not only from the previous layer but also from all preceding layers in the same dense block.

**Transition block** is a component that is used to control the growth of feature maps between dense blocks. DenseNet models consist of dense blocks interspersed with transition modules. The transition module serves the purpose of reducing the spatial dimensions (width and height) of the feature maps while preserving the depth (number of channels) to manage computational efficiency and control overfitting.

Brief explanation:<br>

* Loading pre-trained DenseNet-121 model with ImageNet weights.
* Excluding fully connected classifier layers at the top of the model. This allows you to customize the classifier for your binary classification task.
* Input image dimensions are set to 150x150 pixels with 3 color channels (RGB).
* 9 layers of the base DenseNet-121 model are freezed. This prevents these layers from being updated during training, keeping their pre-trained weights fixed.
* **GlobalAveragePooling2D()** layer is used to compute the spatial average of the feature maps, reducing the spatial dimensions while retaining important features.
* Two dense layers are added with ReLU activation and L2 regularization.
* Dropout rate of 0.2 to help prevent overfitting.
* Final output layer consist of a single neuron and a sigmoid activation function, suitable for binary classification.
* The model is compiled with the **adam** optimizer, which adapts the learning rate during training.
* **binary_crossentropy** is used as the loss function for binary classification.


#### InceptionV3
![Inception drawio](https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/a5abfd35-8a7e-49df-ab73-46869ffa6ca1)

Inception-v3 is a type of convolutional neural network characterized by a depth of 48 layers. It represents an enhanced and refined iteration of the original Inception V1 model. It exhibits reduced computational demands, making it more resource-friendly during training and inference.
<br>
Inception V3 utilizes a series of Inception modules, also known as GoogleNet modules, which are designed to capture features at different spatial resolutions and scales. These modules consist of multiple parallel convolutional layers with different filter sizes (1x1, 3x3, and 5x5) and pooling layers. The idea behind these parallel pathways is to enable the network to learn both fine-grained and higher-level features simultaneously.

* Loading pre-trained **InceptionV3** model with ImageNet weights.
* Omitting the fully connected classifier layers situated at the uppermost section of the model, to adapt and configure the classifier to align precisely with the requirements of binary classification task.
* 9 layers of InceptionV3 model have been immobilized. This action serves to inhibit any updates to these layers during the training process, effectively preserving the predetermined weights that were acquired through pre-training.
* Using GlobalAveragePooling2D layer averages the spatial dimensions of the previous layer's output, reducing the spatial information to a single vector. The Dense layer with 16 units and ReLU activation function further processes this vector.
* Final output layer consist of a single-node dense layer with a sigmoid activation function. In binary classification, a sigmoid activation is commonly used to produce a probability-like output between 0 and 1.
* Using **'adam'** optimizer, the loss function as binary cross-entropy (suitable for binary classification), and the evaluation metric as accuracy.

#### ResNet50



