# Car Classification using Transfer Learning
Solutions to Grab AI For SEA on Computer Vision challenge

This repository contains solution to Grab AI For Sea challenge on Computer Vision. Solutions proposed using Transfer Learning on ResNet50 model implemented on Keras & Tensorflow, and deployed on AzureML.

## Transfer Learning
Transfer learning are one of machine learning technique that focuses on reusing models & architecture that have been pretrained on another dataset. This allows the models to have a base weights and features extraction that have been tuned fairly well on another dataset and can be used to solve another specific problem.

One example of this is by using pretrained models on `ImageNet` dataset, which holds classifiers on many different objects. Using this models, we could use the pretrained layers that have been trained to have better feature extractions, in hope to have more accurate classifications. Thus all we need to do is adding an extra fully connected layer to map the output of the feature to our specific problem outputs.

## Models and Architecture
This repository focuses on training the data on ResNet50 Architecture which is pretrained using the `ImageNet` dataset.

Using the default archicture input and output size will result in 7 x 7 x 2048 dimensions of features. To match the output of the problems (196 classes of labels) can be done by adding a classifier head on top of the models. This can be done by adding Global Average Pooling (GAP) and then applying Dense layer to match the output of the problems statement.

Other parameters used along the ResNet models are Stochastic Gradient Descent (SGD) optimizer and Sparse Categorical Cross Entropy loss function.

## Feature extraction
To feed the training images on the models, this solution uses Image Augmentation (using Keras ImageDataGenerator API), which performs augments & modify the image to create more images with different scenarios. Augmentations used are Horizontal Flip (which flips the image horizontally) and Random Rotation (which rotate the image by 20 degrees max). Image will be resized to match the input models (224, 224, 3) dimension.

## Training Experiments
Training models are done by tuning the hyperparameters, such as epoch & batch_size. Best models are achieved using 32 batch_size, 0.005 learning rate with 0.0001 decay, with 250 x 32 images trained each epochs. Final loss is 4.661 with 93% final accuracy.

## Conclusion
- Due to the time constraints, the solution proposed is not perfect, it still need way more images to be trained each epoch, with another hyperparameter tuning along the way.
- It is also needed to fine tune the models. Maybe unfreeze some of the top convolutional layers to better fit the cars data. This should be proceed with caution as this may cause overfitting.
- Some image preprocessing can be done, such as cropping the image by the bounding box to reduce noise, and changing the aspect ratio of the image.

## Folder Documentation
- `models`  folder holds the implementation of models that are used
- `notebook` folder holds jupyter notebooks that are used for data exploration and experiment regarding the preprocessing, model creation & training
- `outputs` folder holds the best saved models after training
- `remote_run` folder holds AzureML specific code that are used to deploy training to Azure. A slight modification of the code used

## File documentation
- `load_data` contains code for loading the dataframe, extracting the labels from the annotations, and creating generators for data training
- `test` contains code for test the dataframe using the given test set
- `train_resnet` contains code for training the model locally

## References
- [Monza: Image Classification of Vehicle Make and Model Using Convolutional
Neural Networks and Transfer Learning](http://cs231n.stanford.edu/reports/2015/pdfs/lediurfinal.pdf)
