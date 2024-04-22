# COMP6721-GROUPA
# Traffic Sign Detection with CNN Models: README

# Overview:

Traffic signs play a crucial role in managing traffic flow, ensuring road safety, and providing important information to drivers and pedestrians. This project focuses on developing a robust traffic sign recognition system using advanced computer vision techniques, specifically Convolutional Neural Network (CNN) models. The goal is to explore how various hyperparameters and models impact accuracy in traffic sign detection across diverse datasets.

# Datasets:

Three diverse datasets are utilized for this project:

Dataset 1: Contains 3950 images with 10 classes, available in JPG format.
Dataset 2: Comprises 3106 images with 20 classes, available in PNG format.
Dataset 3: Includes 877 images with 4 classes, available in PNG format.
These datasets vary in class counts, image sizes, and formats, posing challenges in training CNN models effectively.
They can be obtained from this link:
https://www.kaggle.com/datasets/pkdarabi/cardetection?resource=download
https://www.kaggle.com/datasets/andrewmvd/road-sign-detection/data
https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification

# CNN Models:

Three CNN architectures are employed in the project:

ResNet 18: A variant of the Residual Network (ResNet) architecture, known for its robustness in image recognition tasks.
ShuffleNet V2: A lightweight CNN architecture designed for efficient image classification on resource-constrained devices.
MobileNet V3: Specifically crafted for efficient inference on mobile and edge devices, offering improved representational capabilities.
Each model is selected based on its efficiency and accuracy across various tasks, particularly when deployed on resource-constrained devices.

# Methodologies:

Data preprocessing techniques such as normalization, tensor transforms, and resizing are applied to enhance model performance.
Hyperparameter tuning is conducted, including experimenting with batch sizes and learning rates to improve convergence stability and prevent overfitting.
Model performance is evaluated using metrics like accuracy, precision, recall, and F1 score, with a preference for F1 score due to its relevance in traffic sign detection.
The datasets are partitioned into train, test, and validation subsets for efficient processing during training and evaluation.
Standardization of input images is ensured by converting them to grayscale with three channels and resizing them to 224 x 224 pixels.
Results:

Each model's performance is evaluated on all three datasets, with varying accuracies observed.
ResNet 18 generally performs better across datasets, followed by MobileNet V3, while ShuffleNet V2 exhibits lower accuracy, potentially due to its compact architecture.
Challenges in dataset handling, including noise, artifacts, and distortions, impact model performance, with ResNet 18 demonstrating better adaptability to diverse datasets compared to ShuffleNet V2 and MobileNet V3.
Future Improvements:

Further analysis of confusion matrices to identify model weaknesses and address them effectively.
Experimentation with different hyperparameter values, loss functions, and transfer learning techniques to enhance model performance.
Introduction of early stopping to prevent overfitting and improve training efficiency.
Continued refinement of the model's performance to adapt to different data patterns.

# Requirements and libraries

The code is run on google colab with most of the dependencies installed already. Any other dependencies are mentioned in the cells on Colab and will be imported after running the cells.
 
# Training and Validation on the Sample Dataset

1. The dataset sample is provided in the zip file named 'Test_Dataset.zip'
2. The sample is taken from https://www.kaggle.com/datasets/andrewmvd/road-sign-detection/data and contains 100 images and the corresponding annotations.
3. The sample is to be used in the codes present in the folder named: Dataset_4_Classes
4. Upload the zip file to your google drive.
5. The folder to be run the code on is 'Dataset1'.
6. The first file to be run is present in folder 'Dataset1/CODE_AI' and named 'Dataset_Split.ipynb'.
7. In the cell [2], relpace the path with the path containing the uploaded dataset.
 ex : !unzip "/path_to_your_dataset" -d "/content/dataset"
8. In cell [5], make sure the path is the path to the folder containing the labels and annotations folder.
 ex : base_dir = '/path_to_file_containing_labels_and_annotations'
9. Use the same path for cell [10]:
 ex : !zip -r /content/dataset1.zip /path_to_file_containing_labels_and_annotations
10. After running the remaining cells, you will get a zip folder downloaded containing the dataset split into train, test and validation datasets
11. After uploading this dataset to the drive, it can be run on the other three codes.
12. In the 'Resnet18_Classes_4.ipynb' file, in cell [4], replace the path with the path to the newly uploaded dataset.
  ex : !unzip "/path_to_the_dataset" -d "/content/dataset"
13.  In cell [6], make sure the path is the path to the folder containing the train, test and val folders.
  ex : dataset_root = '/path_to_file_containing_train_test_and_val'
14. Run the rest of the code, and the graphs for visulaizing will be available after running the last few cells.
