# tensorflow-classifier-examples
Classifier scripts for SVHN and FashionMNIST datasets

# Dependencies
- Python 3.9.2
- matplotlib 3.5.1
- scipy 1.7.3
- scikit-learn  1.0.2
- numpy 1.21.4
- tensorflow 2.7.0

# How to run:
- For FashionMNIST: simply run the script with `python .\FashionMNIST_classifier_example.py` , it should download the dataset, load it in and start training. 
- For SVHN: download train_32x32.mat and test_32x32.mat (Format 2) from (http://ufldl.stanford.edu/housenumbers/). Place the files in same directory as the script. Then run the script.

After running, the scripts should generate 3 "png" files which show the graphs for accuracy, loss and the confusion matrix, respectively.

# Results:
- FashionMNIST test set: Accuracy ~93.3% , ~~Loss 0.22
- SVHN test set: Accuracy ~92.2%, ~~Loss 0.3
