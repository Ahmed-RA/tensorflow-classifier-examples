import os
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import sklearn.metrics as skmt
import tensorflow as tf

#function takes input dataset and labels and outputs indices for the input which represent equal samples per class
#eg: average sample per class is 7000, function performs oversampling for classes under 7000 samples. otherwise draws 
#7000 samples for classes which are over the average
def balance_dataset(x_set, y_set):

    order = np.argsort(y_set, axis=0)
    y_set = np.sort(y_set, axis=0)
    x_set = x_set[order]
    x_set = np.reshape(x_set, (-1,32,32,3))

    samples_per_class = int(len(x_set)/10)

    numOfImgPerLbl = dict()
    for i in range(len(y_set)):
            
        tmp_int = y_set[i]
        tmp_int = tmp_int[0]
        if tmp_int in numOfImgPerLbl:
            numOfImgPerLbl[tmp_int] = numOfImgPerLbl[tmp_int] + 1
        else:
            numOfImgPerLbl[tmp_int] = 1

    indices = list()

    #for loop for oversampling if needed
    counter = 0
    for i in numOfImgPerLbl.keys():
        if numOfImgPerLbl[i] >= samples_per_class:
            sample_indices = np.arange(counter,samples_per_class + counter) 
    
        else:
            sample_indices = np.arange(counter,numOfImgPerLbl[i] + counter - 1) 
            tmp_array = np.random.choice(sample_indices, samples_per_class - len(sample_indices),replace=False)
            sample_indices = np.append(sample_indices, tmp_array, axis=0)
        counter += numOfImgPerLbl[i]
        indices = indices + sample_indices.tolist()

    indices = np.array(indices).astype(np.int32) #for iterability in loops
    np.random.shuffle(indices)

    
    #for debugging purposes, check the samples per class for output indices
    numOfImgPerLbl2 = dict()
    for i in indices:
        tmp_int = y_set[i]
        tmp_int = tmp_int[0]
        if tmp_int in numOfImgPerLbl2:
            numOfImgPerLbl2[tmp_int] = numOfImgPerLbl2[tmp_int] + 1
        else:
            numOfImgPerLbl2[tmp_int] = 1

    return indices


#load svhn data and normalization
def load_normalize():
    

    x_train_loaded = sio.loadmat('train_32x32.mat')
    x_test_loaded = sio.loadmat('test_32x32.mat')

    x_train = x_train_loaded["X"]
    x_train = np.transpose(x_train, (3,0,1,2))

    samples_per_class = int(len(x_train)/10)
    x_train_balanced = np.zeros((samples_per_class*10,32,32,3))

    y_train = x_train_loaded["y"]
    y_train[y_train==10] = 0
    y_train_onehot = np.zeros((samples_per_class*10,10))


    x_test = x_test_loaded["X"]
    x_test = np.transpose(x_test, (3,0,1,2))
    samples_per_class_test = int(len(x_test)/10)
    x_test_balanced = np.zeros((samples_per_class_test*10,32,32,3))

    y_test = x_test_loaded["y"]
    y_test[y_test==10] = 0
    y_test_onehot = np.zeros((samples_per_class_test*10,10))

       


    
    #produce indices for balanced class representation
    indices = balance_dataset(x_train, y_train)
   
    indices_test = balance_dataset(x_test, y_test)


    progress_tmp = 0

    for i in range(len(x_train_balanced)):
                      
        x_train_balanced[i] = normalize_input(x_train[indices[i]]).numpy()
        
        label = y_train[indices[i]]

        y_train_onehot[i][label[0]] = 1

        if i%(int(len(x_train)*0.1))==0:
            progress_tmp += 10
            print("normalizing training images " + str(progress_tmp) + "prc complete")

    
    progress_tmp = 0

    for i in range(len(indices_test)):
        
        x_test_balanced[i] = normalize_input(x_test[indices_test[i]]).numpy()
        
        label = y_test[indices_test[i]]

        y_test_onehot[i][label[0]] = 1

        if i%(int(len(indices_test)*0.1))==0:
            progress_tmp += 10
            print("normalizing test images " + str(progress_tmp) + "prc complete")
    

    #return normalized images as tf vectors
    return tf.constant(x_train_balanced),tf.constant(y_train_onehot),tf.constant(x_test_balanced),tf.constant(y_test_onehot)




def normalize_input(X):
    return tf.math.l2_normalize(tf.cast(X, tf.float32), axis=0)





def keras_train_validate(X,T, Testdata, Testlabel):


    model = tf.keras.Sequential()


    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(32,32,3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu' ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Dropout(0.3))
        
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Dropout(0.3))



    model.add(tf.keras.layers.Flatten())

        

    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))


    model.add(tf.keras.layers.Dense(units=10, activation = 'softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004), 
        loss = tf.keras.losses.CategoricalCrossentropy(), 
        metrics = [tf.keras.metrics.CategoricalAccuracy()]) 
    
    history = model.fit(X, T, epochs = 25, verbose = 1, validation_data = (Testdata,Testlabel)) 

    predictions = model.predict(Testdata)
    predictions_reduced = tf.math.argmax(predictions, 1).numpy()  
    testlabel_reduced = tf.math.argmax(Testlabel, 1).numpy()
        
    conf_matrix = skmt.confusion_matrix(testlabel_reduced, predictions_reduced, labels=[0,1,2,3,4,5,6,7,8,9])
        


    return history, conf_matrix
    


def plot_acc_loss(history, N_samples, conf_matrix):
    f1 = plt.figure()
    plt.plot(history.history['categorical_accuracy'], 'r', label='Training accuracy')
    plt.plot(history.history['val_categorical_accuracy'], 'g', label='Validation accuracy')
    plt.grid(visible=True)
    plt.yticks(np.arange(0.5,1,0.02))
    plt.title('Training Vs Validation Accuracy')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.getcwd() + "\\" + "Accuracy_" + str(N_samples) + ".png") #todo: dynamic paths from cmdline

    f2 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(visible=True)
    plt.yticks(np.arange(0,1,0.05))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.getcwd() + "\\" "Loss_" + str(N_samples) + ".png")

    f3 = skmt.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0,1,2,3,4,5,6,7,8,9])
    f3.plot()

    diag_sum = 0
    for i in range(conf_matrix.shape[0]):
        diag_sum += conf_matrix[i][i]

    print("confusion matrix accuracy is " + str(diag_sum/conf_matrix.sum()))
    print("confusion matrix error is " + str(1 - (diag_sum/conf_matrix.sum())))

    plt.savefig(os.getcwd() + "\\" + "ConfMat_" + str(N_samples) + ".png")







if __name__=="__main__":

    
    N_samples = 73257

    #load data and normalize
    X, T, Test_data, Test_label = load_normalize()
    
    #the fun part
    history, confusion_matrix = keras_train_validate(X,T, Test_data, Test_label)

    plot_acc_loss(history, N_samples, confusion_matrix)

    print("end")




    





        