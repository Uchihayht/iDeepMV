# iDeepMV
iDeepMV is designed to predict which RBPs an unexplored RNA can bind to. It integrates multi-view feature learning, deep feature learning, and multi-label classification technology for RBP recognition. First, based on the raw RNA sequences, we extracted the amino acid sequence view’s data and the dipeptide component view’s data; Then, for the data from different views, we design deep neural network models of the respective views to learn the deep features, and the extracted deep features are further used to train multi-label classifiers that can effectively take the correlation of the labels into account; Finally, the voting mechanism is used to make a multi-view comprehensive decision on the results of each view to further improve the prediction accuracy.