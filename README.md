### **Detection of Cracks on Concrete Surfaces Utilizing Convolutional Neural Networks (CNNs) and OpenCV for Preprocessing**

### Project Overview
The project endeavors to tackle the exigent requirement for an efficient and automated system designed to identify surface cracks across diverse infrastructural and industrial environments. Through the synergistic utilization of OpenCV and Convolutional Neural Networks (CNN), this endeavor aims to transform conventional methodologies of crack detection, presenting an enhanced paradigm that is characterized by heightened accuracy, promptness, and proactive intervention. The amalgamation of OpenCV's prowess in computer vision with CNN's proficiency in deep learning not only streamlines the process of surface crack detection but also furnishes a solution that is both scalable and adaptable to real-world scenarios.

The **goal** of this project is to develop a Deep Learning CNN model capable of effectively detecting the presence of cracks in images of concrete surfaces.

### Data Description and Source

The dataset utilized for this project was obtained from Kaggle, accessible through the link: [accessible here](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection/). This dataset comprises images categorized into two classes: images with cracks and images without cracks. Each class consists of 20,000 images. The images are standardized to 227x227 pixels with RGB channels. This dataset serves as the foundation for training and evaluating the performance of the developed model.


### Process

### Tools

- Machine Learning and Deep Learning Algorithms:
    - Artificial Neural Network (ANN)
    - Convolution Neural Network (CNN)
 

- Programming Language:
   - Python [Download here](https://www.python.org/)

### Data Cleaning/Preparation

Load Images with Cracks (resize to 64x64)
Load Images without Cracks
#Split Image Data from Label Data
## Prepare Data for Training
  #Split into train/test
  #normalize data scaling values from 0-1


### Methods

The methodology adopted to construct a proficient predictive model in this project involved a progressive refinement process, commencing with rudimentary processing techniques and a CNN architecture, subsequently iterated to enhance performance. Initially, optimization was pursued utilizing color images, followed by a transition to grayscale images, culminating in the integration of edge detection methodologies.


Initially, an Artificial Neural Network (ANN) was applied to the dataset as a benchmark for comparison with the subsequent CNN model. The process commenced with the implementation of a basic CNN architecture featuring one convolutional layer and two connected layers. Subsequently, the architecture evolved to incorporate a more complex CNN model comprising four convolutional layers and two connected layers. Furthermore, optimization was achieved through the utilization of Canny Images for enhanced performance evaluation.


#### The methods outlined in the attached Jupyter notebook are delineated as follows:
1. Color CNNConcreteCrackIdentification

The dataset consists of color images of concrete surfaces, with positive samples containing cracks and negative samples without cracks. The executed code follows these steps:
  - Reads and resizes the images, generating arrays for both cracked and non-cracked data separately.
  - Concatenates the data and labels for subsequent processing.
  - Divides the dataset into training and testing subsets.
  - Normalizes pixel values to the interval [0, 1].
  - Prepares textual descriptions for target labels.
  - Segregates the data into training and testing sets.
  - Establishes and trains a rudimentary CNN architecture featuring one convolutional layer and two connected layers.
  - Iteratively constructs and trains CNN models with varying numbers of convolutional and connected layers.
  - Assesses and illustrates each model's performance using confusion matrices and classification reports.
  - Identifies the CNN model with four convolutional layers and three connected layers as optimal.
  - Implements early stopping and model checkpoint callbacks for optimization purposes.
  - Trains the model and assesses its performance on validation and test sets.
  - The code explores different CNN architectures and optimization strategies aimed at enhancing model performance.


2. Grayscale CNNConcreteCrackIdentification
The executed code follows these steps:

The provided code executes the following tasks:

  - Reads grayscale images depicting concrete surfaces with and without cracks.
  - Resizes the images to dimensions of 64x64 pixels.
  - Divides the data into training and testing partitions.
  - Normalizes pixel values to the interval [0, 1].
  - Defines and compiles CNN models featuring various architectures, ranging from one to four convolutional layers.
  - Trains each CNN model using the training data.
  - Assesses the performance of each CNN model on the test set.
  - Generates confusion matrices and classification reports for each CNN model.
  - Identifies the CNN model with four convolutional layers and two connected layers for further optimization.
  - Splits the training data into training and validation sets.
  - Implements early stopping and model checkpoint callbacks during the training process.




3. Grayscale with Edge Detection CNNConcreteCrackIdentification

The provided code accomplishes the following tasks:

  - Loads images containing cracks, resizes them to 64x64 pixels, and adds them to the "cracks_data" array.
  - Loads images without cracks, resizes them similarly, and appends them to the "no_cracks_data" array.
  - Combines image data and corresponding labels for both cracked and non-cracked images.
  - Divides the data into training and testing sets.
  - Defines a function to conduct Sobel and Canny edge detection on an image and displays the resulting edges.
  - Applies edge detection to a subset of sample images.
- Prepares Data for Training using Sobel Intensity Images:
  - Applies Sobel edge detection to all images and stores the results in "sobel_img_data".
  - Segregates the data into training and testing sets, normalizes pixel values, and prepares it for training.
  - Constructs a convolutional neural network (CNN) model comprising four convolutional layers, two dense layers, and utilizes Sobel intensity images for training.
  - Trains the model, validates its performance, and saves the best model based on validation accuracy.
  - Evaluates the model on both validation and test sets.
- Prepares Data for Training using Canny Images:
  - Applies Canny edge detection to all images and stores the results in "canny_img_data".
  - Splits the data into training and testing subsets, normalizes pixel values, and readies it for training.
  - Defines a new CNN model with the same architecture as before, utilizing Canny edge images for training.
  - Trains the model, validates its performance, and saves the best model based on validation accuracy.
  - Evaluates the model on both validation and test sets.



4. Application of Best Model to Images/Videos outside of Train/Test Dataset
     
The provided code facilitates concrete crack classification utilizing a trained neural network model based on Sobel edge detection, capable of processing various input types such as images and videos. The code execution comprises the following steps:

  - Imports the previously trained and saved best Sobel-based model.
  - Defines functions for acquiring file paths and extracting frames from video files.
  - Implements a concrete crack classification function that prompts user input to specify the input type (image, video, or batches of images/videos).
  - For image inputs, the function reads and resizes the images, applies Sobel edge detection, normalizes the data, and makes predictions using the loaded model.
  - For video inputs, frames are extracted, Sobel edge detection is applied to each frame, data is normalized, and predictions are made using the loaded model.
  - The function outputs predictions indicating the presence or absence of concrete cracks.
  - This comprehensive approach enables seamless classification of concrete cracks across diverse input formats, offering flexibility and adaptability for real-world applications.


### Results/Findings
The following figures and table show confusion matrices for best models from each explored processing method and comparison of their accuracy scores:

<img src = "images/ColorImages.png">
<img src = "images/GrayScaleImages.png">
<img src = "images/SobelEdgeDetectionImages.png">
<img src = "images/GrayScaleImages.png">


Comparing the various model's accuracy scores, it is evident that using grayscale or sobel-edgedetection provide optimum results. Since many of the applications of a model like this would be to identify cracks that need to be addressed, we will make the assumption that it is of more benefit to increase the number of correctly identified cracks, even if this means an increase of falsely identified cracks. For this reason, we will select the Sobel-EdgeDetection model as the optimum model in this project.
Applying the Sobel-EdgeDetection trained model to some external images and videos resulted in the following:
•
Six images were used (these images are cropped from images taken of concrete cylinders used as compressive strength samples), 3 with cracks present and 3 without.
•
Of the six images, the model predicted all 6 correctly as seen in Figure 5.
•
Four short video clips were used (these videos are cropped from videos taken of sidewalk pathways), 2 with cracks present and 2 without.
•
Of the four videos, the model correctly predicted both videos without cracks and 1 of the videos with cracks present as seen in Figure 6.
