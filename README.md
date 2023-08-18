# Discriminating between Chihuahuas and Muffins: A Deep Learning Approach

## Introduction

Throughout the 2010s, the field of deep learning made remarkable strides, particularly in addressing anticipated tasks like object classification, speech recognition, text analysis, image synthesis, and more. Notably, AI competitions such as the "ImageNet challenge" played a pivotal role in catapulting convolutional architectures into the limelight. These architectures gained immense popularity for their effectiveness in resolving object recognition and classification dilemmas.

In the field of AI and computer vision, we address the challenge of distinguishing between objects that share close similarities, such as Chihuahuas and muffins.

![1*bt-E2YcPafjiPbZFDMMmNQ](https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/4c333fc3-e555-4c2e-9327-04492e69f1d3)

The Chihuahua vs. Muffin Challenge in the field of AI/ML introduces a captivating test of image classification algorithms. This unique challenge centers on differentiating between images of Chihuahua dogs and muffins, two subjects that can surprisingly share visual similarities.

## Getting Started

#### Prerequisites

* Check if python environment exist.
```console
python3 --version
```

* Install tensorflow
```console
python3 -m pip install tensorflow
```

* Verify tensorflow installation
```console
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

* Install Keras
```console
numpy matplotlib keras
```
* Guide to install [python](https://www.python.org/downloads/) and [tensorflow](https://www.tensorflow.org/install/pip#macos).

* Guide to install [anaconda](https://www.anaconda.com/).

GPU Integrated notebooks.
* [Kaggle](https://www.kaggle.com/) 
* [GoogleColab](https://colab.google)

## How to run

**Setup Dataset**
* Download [Chihuahua vs Muffin]((https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification)) Dataset.
* The dataset consists of 2 folders train and test.
* Both train and test folder contains 2 folder named as chihuahua and muffin.
* The name of these 2 folders are considered as label for the dataset.

**Code Run**
* Run **project.ipynb** using [anaconda](https://www.anaconda.com/download), or any other GPU integrated notebooks (eg: [Kaggle](https://www.kaggle.com/), [GoogleColab](https://colab.google/)).
* If using any GPU integrated notebooks first upload these dataset and then run the code.


## Dataset

The dataset consists of 5917 images, serving as the basis for our in-depth investigation into similar image recognition. 

* **Training and Validation set(80%):** With 4733 images, the training set assumes a crucial role in cultivating the AI's comprehension. This particular subset captures subtle intricacies in patterns and textures, imparting a holistic grasp of Chihuahuas and muffins to the model.
* **Test set(20%):** Comprising 1184 images, the testing set holds paramount importance in our evaluation process. Through assessing the model's performance on this distinct subset, we attain a dependable gauge of its practical capabilities in real-world scenarios.


## Methodology

### Data Augmetation
* **ImageDataGenerator** class from the Keras library is used to perform data augmentation. The images are being resized to a target size of 150x150 pixels.

* **Rotation Range:** Images are rotated by an angle of 40. This helps the model become invariant to rotation.

* **Width and Height Shift Range:** The images are shifted horizontally and vertically by a proportions of 0.2. This simulates changes in perspective and object positioning.

* **Zoom Range:** Images are zoomed in and out by a factors of 0.2. This helps the model learn to recognize objects at different scales.

* **Horizontal Flip:** Images are flipped horizontally with a certain probability. This helps the model become invariant to horizontal flips.

* **Rescale:** The pixel values of the images are rescaled to a range of [0, 1] by dividing them by 255. This standardizes the pixel values.

### Early Stopping
* In order to mitigate the tendency of models to overfit, the strategy of early stopping has been employed. Early stopping entails the vigilant tracking of the model's performance on a distinct validation dataset throughout the training phase. The training process is curtailed as soon as indications of deteriorating performance on the validation dataset become evident.

### CNN Architecture
#### Simple CNN
![Shallow Network drawio](https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/370c305b-8d00-44ea-852c-70bd5ae42e0e)

A shallow CNN model refers to a Convolutional Neural Network (CNN) architecture with a relatively small number of layers. Shallow CNN models are often used for simpler image recognition tasks or as baseline models for comparison with more complex architectures.

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

![Resnet50 drawio](https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/fab0e116-9f28-4b4f-aa89-3233f9ca4b3f)

ResNet-50, short for "Residual Network 50," introduces residual blocks. These blocks allow the network to learn residual functions, which are the differences between the desired output and the current approximation produced by the network. This is achieved through the use of shortcut connections, also known as skip connections or identity mappings.

* Loading pre-trained **Resnet50** model with ImageNet weights.
* Removing the topmost fully connected classifier layers of the model and customizing the classifier to precisely suit the needs of a binary classification task.
* The required input_shape is set to 1150x150 pixels with 3 RGB to the input image dimensions.
* 35 layers are freezed, meaning they won't be updated during training, helping to retain the pre-trained knowledge from ImageNet.
* The Flatten layer is used to flatten the output tensor from the base model.
* Two Dense layers with ReLU activations are added to introduce non-linearity. These layers can help the model learn complex patterns from the flattened features.
* The final Dense layer with a sigmoid activation produces a single output, which is common for binary classification tasks.



## Results

<table border="1">
  <tr>
    <td><p>Shallow CNN Model</p></td>
    <td>VGG19</td>
  </tr>
  <tr>
    <td><img src='https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/3e9fc730-25fa-4bba-a91e-b61a3dbff217' width='350' height='250'/>     </td>
    <td><img src='https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/657073c4-3c76-4473-9b3e-510654a38428' width='350' height='250'/> </td>
  </tr>
</table>

<table border="1">

  <tr>
    <td><p>DenseNet121</p></td>
    <td>InceptionV3</td>
  </tr>
  <tr>
    <td><img src='https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/fbfd6df5-1ad7-4a8b-ac0d-f5fe7027ecbe' width='350' height='250'/>     </td>
    <td><img src='https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/6cb536c4-c3e2-40cd-a928-40695ddcac4d' width='350' height='250'/>     </td>
  </tr>
</table>

<table border="1">
  <tr>
    <td><p>Resnet50</p></td>
  </tr>
  <tr>
    <td><img src='https://github.com/DynDevelopers/CryptoDashboard/assets/42007119/0dd319c8-ce6f-4b57-a40f-a22c81dbacc6' width='350' height='250'/>     </td>
  </tr>
</table>



## Future Scope

The examination covered a subset comprising 4733 images. It's crucial to recognize that the accuracy percentages attained within this particular context should not be universally generalized. The conclusions drawn from such analyses are substantially impacted by factors like dataset instance size, composition, computational resources, and other variables. Pretraining models on an extensive dataset and subsequently fine-tune it using the target dataset for the purpose of image recognition. Using a learning rate scheduler that will dynamically adjusts the learning rate during the training process, thus augmenting the optimization procedure.

## References
* [Kaggle Dataset](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification)
* Andrej Karpathy et al. “Convolutional neural networks for visual recognition”. In: Notes accompany the Stanford CS class CS231 (2017).
* Ramaprasad Poojary, Roma Raina, and Amit Kumar Mondal. “Effect of data-augmentation on fine-tuned CNN model performance”. In: IAES International
* Journal of Artificial Intelligence 10.1 (2021), p. 84.
[3] Enkhtogtokh Togootogtokh and Amarzaya Amartuvshin. “Deep learning approach for very similar objects recognition application on chihuahua and muffin problem”. In: arXiv preprint arXiv:1801.09573 (2018).
