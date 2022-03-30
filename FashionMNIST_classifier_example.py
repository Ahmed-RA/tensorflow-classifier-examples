import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skmt
import tensorflow as tf





def pad_and_normalize():
    
    #load fashionmnist data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    #check for number of images per class on training and testing sets
    numOfImgPerLbl = dict()
    for i in range(len(y_train)):
            
        tmp_str = y_train[i]
        if tmp_str in numOfImgPerLbl:
            numOfImgPerLbl[tmp_str] = numOfImgPerLbl[tmp_str] + 1
        else:
            numOfImgPerLbl[tmp_str] = 1

    numOfImgPerLbl_test = dict()
    for i in range(len(y_test)):
            
        tmp_str = y_test[i]
        if tmp_str in numOfImgPerLbl_test:
            numOfImgPerLbl_test[tmp_str] = numOfImgPerLbl_test[tmp_str] + 1
        else:
            numOfImgPerLbl_test[tmp_str] = 1

    
    #initial numpy matrices
    X_padded_train = np.zeros((len(x_train), 32,32,1))
    X_padded_test = np.zeros((len(x_test), 32,32,1))

    y_train_onehot = np.zeros((len(y_train), 10))
    y_test_onehot = np.zeros((len(y_test), 10))



    progress_tmp = 0

    for i in range(len(X_padded_train)):
        
        #pad 28*28 into 32*32
       
        image = np.pad(x_train[i], ((2,2),(2,2)), 'constant')
              
        img_vector = normalize_input(image)
        
        img_matrix = img_vector.numpy()
        
        X_padded_train[i] = np.reshape(img_matrix, (32,32,1))

        y_train_onehot[i][y_train[i]] = 1

        if i%(len(X_padded_train)*0.1)==0:
            progress_tmp += 10
            print("padding training images " + str(progress_tmp) + "prc complete")

    
    progress_tmp = 0

    for i in range(len(X_padded_test)):
        
        #pad 28*28 into 32*32
       
        image = np.pad(x_test[i], ((2,2),(2,2)), 'constant')
              
        img_vector = normalize_input(image)
        
        img_matrix = img_vector.numpy()
        
        X_padded_test[i] = np.reshape(img_matrix, (32,32,1))

        y_test_onehot[i][y_test[i]] = 1

        if i%(len(X_padded_test)*0.1)==0:
            progress_tmp += 10
            print("padding testing images " + str(progress_tmp) + "prc complete")
    

    #return padded images as tf vectors
    return tf.constant(X_padded_train),tf.constant(y_train_onehot),tf.constant(X_padded_test),tf.constant(y_test_onehot)




def normalize_input(X):
    return tf.math.l2_normalize(tf.cast(X, tf.float32), axis=0)





def keras_train_validate(X,T, Testdata, Testlabel):


    model = tf.keras.Sequential()


    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(32,32,1)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))


    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation("softmax"))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss = tf.keras.losses.CategoricalCrossentropy(), 
        metrics = [tf.keras.metrics.CategoricalAccuracy()]) 
    
    history = model.fit(X, T, epochs = 25, verbose = 1, batch_size=32, validation_data = (Testdata,Testlabel)) 

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

    
    N_samples = 60000

    #load data, pad and normalize
    X, T, Test_data, Test_label = pad_and_normalize()
    
    #the fun part
    history, confusion_matrix = keras_train_validate(X,T, Test_data, Test_label)

    plot_acc_loss(history, N_samples, confusion_matrix)

    print("end")




    





        