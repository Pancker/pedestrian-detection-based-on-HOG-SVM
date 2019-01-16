# pedestrian-detection-based-on-HOG-SVM
A classical method to implement pedestrian detection. 

The content of this pedestrian detection is divided into the following processes:
1. Prepare training sample set: including positive sample set and negative sample set.
2. Processing of training samples
3. HOG feature extraction of positive samples.
4. HOG feature of negative sample is extracted.
5. Mark positive and negative samples, positive sample is 1, negative sample is -1.
6. HOG feature of positive and negative samples and label of positive and negative samples are input into SVM for training.
7. SVM results after training are used as a classifier for HOG multi-scale detection
8. Conduct pedestrian detection and non-maximum suppression, and draw a rectangle


The positive dataset I used was MIT and CUHK_Person_Reidentification.
The negative dataset I used were many small screenshoots of my school playground.
