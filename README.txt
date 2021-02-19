==============================================================================================
==============================================================================================
This file describes the steps to run the assignment2_COMP90049 ipynb file 
COMP90049: Introduction to Machine Learning
Project 2: Music Genre Prediction from Audio, Metadata, and Lyric Features!

=========================================
Helper functions used for the assignments
=========================================

* reads the csv files and imports the data ina  dataframe
* removes track id from labels before processing
** read_files()

* splits features for a given column range
** split_features(features, column1, column2)
    
* converts a textual feature column into tfidf vector
** text_features_tfidf(train_features,valid_features,test_features,text_column)

* converts string labels into numveric values
** convert_labels_numeric(train_labels,valid_labels)

* scales values between 0 and 1
** min_max_scaler(train_features,valid_features,test_features)
   
* trains a classifier for a given train feature set and train labels
** train(train_features,train_labels,classifier)

* predicts values for a classifier for a given feature set
** predict(valid_features,trained_classifier)

* evaluates accuracy score for a set of predicted labels
** evaluate(actual_values,predicted_values)
  
* plots a bar graph showing the count for each type of genre 
** plot_frequecy_labels(train_labels,label_encoder)

* plots a multi bar graph showing the counts for each type of genre for different classifiers
** plot_predicted_actual_labels(predicted_DT_labels,predicted_MLP_labels,predicted_DC_labels,valid_labels)
    
    
* plots a dataframe for the accuracy score for each type of classifier
** plot_predicted_accuracy(predicted_DT_accuracy,predicted_MLP_accuracy,predicted_DC_accuracy)
   
* initialises the decision tree classifier with hyperparameters for tuning
** decision_tree_classifier()
    
* initialises the multilayer perceptron with hyperparameters for tuning
** multi_layer_perceptron_classifier()
   
* initialises the dummy classifier
** dummy_classifier():
    
* selects features recursively using the RFECV function based on a tuned classifier
** select_features(classifier,train_features,valid_features,test_features,train_labels):
    
* tunes a classifier based on a given set of hyper parameters
** train_GSCV_classifier(train_features,train_labels,classifier,parameter):
** Here we have set n_jobs=2 to run two jobs concurrently in parallel in order to reduce time
** However, this function still takes around 25 mins

===============================
Steps to run for the experiment
===============================

* There are 11 blocks in the code after the helper function code
* Each block represents a single step
* To run this experiment all you have to do is to run the blocks sequentially ~
* Some steps take time

* The code first reads the csv files, extract the values, split them into different types of feature sets and scales them between 0 and 1
* The hyperparameters for the classifiers are first tunes and then they are used for training, predicting
* Accuracy scores are calculated and graphs are plotted
* In the end, the predicted values of test dataset are converted to csv

=====
Steps
=====
* Block 1
Used for initialising the classifiers, reading files from csv and removing track id from class labels

* Block 2
Used for spliting audio vectors from the training, validation and testing dataset. Scales the values between 0 and 1. Selects the best number of features for audio feature set. Plots a graph and prints the vector dimensions. - Takes time

* Block 3
User for fine tuning the hyper parameter of the classifiers - Takes time

* Block 4
Used for training the classifier, predicting values and calculating accuracy score for audio data - Takes time

* Block 5
Converts tags to TFIDF vector, scales the values and prints out the dimensions

* Block 6
Used for training the classifier, predicting values and calculating accuracy score for lyrical data - Takes time

* Block 7
Converts title to TFIDF vector, selects the best number of metadata features scales the values and prints out the dimensions

* Block 8
Combine metadata values and print the feature vector dimensions

* Block 9
Used for training the classifier, predicting values and calculating accuracy score for metadata - Takes time

* Block 10
Plots a bar graph showing the count for each type of genre

* Block 11
Used for training the classifier, predicting values and calculating accuracy score for best feature set from test data
Also outputs the predicted values to csv file


    
