# CODTECHINTRENSHIP-task4

__Name:__ MAJJIGA MEGHANA MADHURI

**Company:** CODTECH IT SOLUTIONS

**ID:**  CT6WDS1400

**Domain:** ARTIFICAL INTELLIGENCE

**Duration:**  from JULY 20th, 2024 to SEPTEMBER 5th, 2024.

**Mentor:** MUZAMMIL AHMED

**Overview**

This code provides the end-to-end pipeline of solving an image classification task on the CIFAR-10 dataset by using a Convolutional Neural Network. This is achieved by the following steps;

**Import of Libraries:** Importing of important libraries like TensorFlow for building and training of a neural network, OpenCV for image processing, and Matplotlib for showing results.

**Loading and Preprocessing the Data:**

It loads the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. It normalizes the images by scaling the pixel values between 0 and 1 to help the neural network converge. Data Split: The training data is split into training and validation datasets using an 80-20 split. This allows the model to train on one part and be validated on the other.

**Model Definition:**

It defines a CNN composed of multiple layers: the Convolutional Layers, or Conv2D, for feature extraction from images; the Max Pooling Layer, or MaxPooling2D, which downsamples the feature maps to reduce the spatial dimensions; the Flatten Layer, which turns the 2D feature maps into a 1D vector; Dense Layers, or fully connected, that map the extracted features onto output classes; Dropout Layer, which prevents overfitting by randomly dropping units during training; and an Output Layer, with a softmax activation function, that assigns probabilities to each class.

**Model Compilation:** 

The model has been compiled using the Adam optimizer with sparse categorical crossentropy as the loss, which can be used for multi-class classification problems.

**Model Training:** 
The model needs to be trained on 10 epochs, and both training and validation accuracy should be tracked for possible future improvements in model performance.

**Model Evaluation:**
The performance evaluation of the model will be conducted on the test dataset. This shows the accuracy score, which essentially denotes the goodness of the model in generalizing to new unseen data.

**Visualization:**
It plots training and validation accuracy vs. number of epochs and shows the improvement of the model.


**Conclusion**
The project successfully takes one through an image classification problem using a CNN. After training the model on the CIFAR-10 dataset, it reaches a certain level of accuracy that is then evaluated on a separate test dataset. In regard to this, the use of a CNN is quite key, since it has the great ability to capture spatial hierarchies in images-a factor attributing to its great effectiveness while executing various tasks in computer vision, including image classification.

The training and validation accuracy curve helps in understanding not only how well the model learns over time but also its tendency to overfit or underfit the data. This workflow provides a good backbone on which complex computer vision tasks-such as object detection, segmentation, or more sophisticated models-can be realized.


**OUTPUT**

![image](https://github.com/user-attachments/assets/32e0080f-36bd-499a-b304-b79b002eda50)
