import numpy as np
import pickle
import tensorflow as tf
from keras import models, layers, 


"""
General Overview of data format:

11 Modulation Types each at 20 different signal-to-noise ratio, where each sample has a vector size of 256, split into
128 in-phase and 128 quadrature components (Where these refer to two sinusoid's of same frequency and that are 90 
degrees out of phase, that is a sine and cosine wave), there is a total of 1000 of these array size 128.

For example; dictionary key which pertains to the modulation type and signal to noise ratio values will be mapped to 
500 total of array 256 sized where 128 are the in-phase components and 128 are the quadrature components

Therefore, in total with 11 (No of modulation types) * 20 (20 different snr values) * 128 (size of each components) * 2 (how many components there are) * 1000 (total amount of values of components)
 = 56320000 total values
 and input samples = 11 * 20 * 1000 = 220000

 1000 arrays per modulation type and its respective signal-to-noise ratio.

 Therefore to classify is to figure out which modulation type and snr each 1000 arrays is mapped to.

#print(*dataset['QPSK', 2])

['QPSK', 2] :[[[-5.58401039e-03 -1.44638738e-03  4.57739085e-03  5.91279473e-03             [9.26337205e-03  6.32310193e-03  2.42765830e-03  5.90267824e-03
                -5.95544651e-03 -2.04047211e-03 -8.02253280e-03 -4.37167333e-03             -4.86804359e-03  3.16158403e-05  3.67330201e-03 -2.64918478e-03
                 9.68801510e-03 -4.29365813e-04 -5.65820886e-03 -1.56519774e-04              3.67798330e-03 -5.12399385e-03 -3.35781695e-03  4.38169902e-03
                 9.18083359e-03 -8.17031693e-03 -3.32616130e-03 -3.83706181e-03              7.23025156e-03  1.51032312e-02 -3.86116421e-03  3.60218529e-03
                 2.73792585e-03 -8.87676049e-03  1.34362755e-02  8.48278962e-03             -9.51823895e-04  5.54428995e-03  1.26734599e-02 -9.04008560e-03
                -8.49103089e-03  1.20306271e-03  8.70861858e-03  2.74055661e-03              1.19866803e-03  1.71840470e-03  3.16794589e-03 -9.26620513e-03
                 4.89505101e-03  4.04845830e-03 -2.41991016e-04 -2.56560580e-03              6.82127487e-04  2.38042980e-04 -6.98410533e-03 -7.44014466e-03
                 3.55843280e-04 -3.60339624e-03  1.22411599e-04  5.25716052e-04              2.96051824e-03 -1.28793903e-02  2.41364562e-03  1.54666463e-03
                -5.86867658e-03 -5.23684220e-03 -3.55379726e-03  7.85416982e-04              2.50411010e-03 -6.39274775e-04  2.06536776e-03  1.07192784e-03
                -1.20433085e-02 -7.49130314e-03  6.36015600e-03 -6.34977873e-03              9.78144910e-03 -2.36725528e-03 -5.25449635e-03 -2.07151147e-03
                -3.25542386e-03 -6.92803366e-03  8.06394892e-05  7.07363850e-03              9.58657067e-04 -1.12365093e-03  6.35598553e-03 -5.16045559e-03
                 2.37252883e-04  2.19245162e-03 -1.12451009e-04  6.38831919e-03             -6.72402466e-03 -9.92396381e-03 -5.09300921e-03 -8.14018492e-03
                 7.21993111e-03  2.62756413e-03  6.98318356e-04  1.04795527e-02              1.62706582e-03  5.72435465e-03 -1.51183782e-03  1.72276758e-02
                -4.54280712e-03 -1.61364453e-03  8.97876266e-03  4.74766502e-03              6.59239944e-03  1.22039383e-02  9.29343235e-03 -1.01398490e-03
                -5.34521649e-03 -4.56042122e-03 -4.25275182e-03 -2.14294670e-03              6.52232487e-03  4.82049910e-03 -3.93229479e-04 -2.48908740e-03
                 1.24218338e-03 -2.91716331e-03 -5.02848485e-03  2.61101290e-03              2.14377884e-03  1.20980130e-03 -7.92906899e-03 -1.64318626e-04        
                 4.34444402e-04  3.44594312e-03  1.92429719e-03  3.76734824e-04             -3.95924039e-03  8.05266295e-03  2.55630678e-03 -1.42084982e-03
                 9.07383393e-03  1.62037101e-03  1.79680483e-03  6.44263532e-03             -1.21878218e-02  4.02815128e-03 -4.13294183e-03 -7.60551123e-03
                -6.29950664e-04  1.26093682e-02  3.45904171e-03 -1.17933452e-02              2.04335010e-04  5.11903316e-03 -4.85667586e-03 -2.41500605e-03
                 5.11476258e-03 -1.05265910e-02  3.65055003e-03 -1.22644473e-04              8.72036628e-03  1.57410260e-02  1.02230720e-02  1.23430602e-02
                -1.18740858e-03 -2.20584287e-03 -4.86190105e-03 -7.74104055e-03             -1.62384775e-03  3.50630027e-03  8.81469715e-03  3.17534024e-04
                -1.13753369e-02 -2.52028555e-03 -5.40231680e-03 -1.07841967e-02             -4.23061242e-03 -7.46789621e-04  9.20665625e-04 -5.17984328e-04
                 4.94938390e-03 -5.49336523e-03 -4.87153957e-05  2.83753709e-03              4.50664852e-03  1.39947026e-03 -1.26646878e-02 -3.55927064e-03
                -8.51597637e-03  6.66276412e-03  3.60417389e-03  4.59533854e-04             -4.98852110e-04  5.61951380e-03 -2.54999194e-03  7.25704618e-03
                 1.47640621e-02  9.51905362e-03  5.63115953e-03  8.15430377e-03              9.37073771e-03 -6.47906633e-03  3.56399338e-03  6.92746742e-03
                 1.05095087e-02  1.29917008e-03  4.34858585e-03 -5.82342921e-03             -4.72967327e-03 -1.43024344e-02  4.49528219e-03 -8.60740524e-03
                -4.77133831e-03  4.84300451e-03 -5.39009739e-03  4.42959368e-03             -8.67190585e-03 -7.91994482e-03 -8.69230554e-03 -3.61651625e-03
                 1.13355126e-02  6.50491519e-03  2.88330601e-03  7.94661511e-03              5.91185549e-03 -1.50937866e-03  6.04824722e-03 -8.40493944e-03
                 8.96142609e-03 -4.34405636e-03  3.59805231e-03  1.01398313e-02              2.25435346e-04  1.38457473e-02  1.13582658e-02 -3.04110465e-04
                 7.13022426e-04  2.47485982e-03 -5.47889713e-03 -6.09123847e-03              6.38689264e-04 -5.36400452e-03  1.04513634e-02 -1.13080069e-03
                -7.16688018e-03  3.76094435e-03 -5.88832470e-03  7.44403340e-04             -2.26811925e-03  7.61763379e-03  1.29662605e-03  6.80402340e-03
                -7.38550443e-03 -4.93408472e-04  7.54382403e-04 -4.46916092e-03]             2.72199605e-03 -1.07705714e-02 -4.11019195e-04 -8.97216960e-04]]

 x 1000           
                ...]


"""
# ================================================== Loading Data ==================================================== #
# Dataset is a dictionary
dataset = pickle.load(open("RML2016.10a_dict.pkl", "rb"),
                      encoding = "bytes")  # encoding = "latin1" or encoding = "bytes", sources say bytes are better https://stackoverflow.com/a/47882939
snr_S, mod_S = map(lambda j : sorted(list(set(map(lambda x : x[j], dataset.keys())))), [1, 0])

dataset_Values = []
dataset_Keys = []

for mod in mod_S :
    for snr in snr_S :
        dataset_Values.append(dataset[(mod, snr)])
        for i in range(len(dataset[(mod, snr)])) :
            dataset_Keys.append((mod, snr))

x_Input = np.vstack(dataset_Values)

"""
Obtains the snr, modulation types and the inputs such that the elements of both list match in terms of the modulated 
signal and its modulation type.

The np.vstack is to "stack" the arrays such that the len of both lists match. Originally dataset_Values was length 220 
with embedded arrays of size 1000, now its just a list of length 220000. 

print(len(dataset_Values)) = 220
print(len(x_Input)) = 220000
print(len(dataset_Keys)) = 220000

"""

# =============================================== Partition the data ================================================= #

np.random.seed(2022)
no_Samples = 220000
no_Train = int(no_Samples / 2)

train_Index = np.random.choice(range(0, no_Samples), size = no_Train, replace = False)
test_Index = list(set(range(0, no_Samples)) - set(train_Index))

x_Train = x_Input[train_Index]
x_Test = x_Input[test_Index]

label_Train = tf.keras.utils.to_categorical(list(map(lambda x : mod_S.index(dataset_Keys[x][0]), train_Index)))
label_Test = tf.keras.utils.to_categorical(list(map(lambda x : mod_S.index(dataset_Keys[x][0]), test_Index)))

"""
np.random.seed(input_Value) = will mean all random numbers will be predictable, in such a way that whe the seed of 2022
is referenced again, the same random numbers will appear.

Data is split equally into training and testing, the train_Index takes the index of all data that needs to be used to
train and test_Index will have the remainder of the indexes.

x_Train and label_Train
x_Test and label_Test

The to_categorical function is to change the list into a onehot vector to use categorical_crossentropy

"""

input_Shape = list(x_Train.shape[1 :])
# print(x_Train.shape, input_Shape)


# ==================================== Build Sequential Model =================================================== #

# Build VT-CNN2 Neural Net model using TF 2.0 --
dropout_rate = 0.5  # dropout rate (%)

# Model build start of a sequential model
model = models.Sequential()
model.add(layers.Reshape([1] + input_Shape, input_shape = input_Shape,
                         name = "first_reshape"))  # Reshapes the model into target shape, if first layer of model, uses input_shape. as the model to turn into target shape Reshape into 2x128x1 HxWxC; CxHxW in output shape
model.add(
    layers.ZeroPadding2D((0, 2), data_format = "channels_first", name = "first_zero_padding"))  # Adds 2 on either side
model.add(layers.Conv2D(256, (1, 3), padding = "valid", activation = "relu", name = "first_convolution_layer",
                        data_format = "channels_first",
                        kernel_initializer = "glorot_uniform"))  # Convolution layer with 256 filters with kernel size (1,3), no padding, data format is # CxHxW kernel initializer refers to what distribution to initialise the weights with, in this case it will be glorot_uniform
model.add(layers.Dropout(dropout_rate,
                         name = "first_dropout"))  # Dropout layer helps to prevent over-fitting i.e. when the model is fits against the training data perfectly, this makes it hard for it decipher new data
model.add(layers.ZeroPadding2D((0, 2), data_format = "channels_first", name = "second_zero_padding"))
model.add(layers.Conv2D(80, (2, 3), padding = "valid", activation = "relu", name = "second_convolution_layer",
                        data_format = "channels_first", kernel_initializer = "glorot_uniform"))
model.add(layers.Dropout(dropout_rate, name = "second_dropout"))
model.add(layers.Flatten())  # Flatten i.e. makes the grid 1 dimensional, multiplies all values
model.add(layers.Dense(256, activation = "relu", kernel_initializer = "he_normal", name = "first_dense_layer"))
model.add(layers.Dropout(dropout_rate, name = "third_dropout"))
model.add(
    layers.Dense(len(mod_S), kernel_initializer = "he_normal", name = "second_dense_layer"))  # No activation applied
model.add(layers.Activation("softmax",
                            name = "softmax_function"))  # Softmax function used as the last layer to normalize the output
model.add(layers.Reshape([len(mod_S)], name = "second_reshape"))
model.compile(loss = tf.keras.losses.CategoricalCrossentropy(), optimizer = "adam", metrics = ["accuracy"])
model.summary()

"""
The VT-CNN2 model is shown below
In the paper figure 1 shows its dimensions as Height x Width x Channel

Data format if channels_first will be Channel x Height x Width and this is shown in output summary as well. 
The first data point is batch size

# 2 x 128 x 1
# Zero Padding (0, 2) filter
# 2 x 132 x 1
# Conv2d 256  (1, 3) filter
# 2 x 130 x 256
# Dropout and Zero Padding 0,2 filter
# 2 x 134 x 256 
# Conv2d 80 (2, 3) filter
# 1 x 132 x 80
# Dropout and Flattening 
# 10560 x 1
# Dense 
# 256 x 1
# Dropout and Dense 
# 11 x 1
# Softmax layer
# 11 x 1

Output: 

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 first_reshape (Reshape)     (None, 1, 2, 128)         0         

 first_zero_padding (ZeroPad  (None, 1, 2, 132)        0         
 ding2D)                                                         

 first_convolution_layer (Co  (None, 256, 2, 130)      1024      
 nv2D)                                                           

 first_dropout (Dropout)     (None, 256, 2, 130)       0         

 second_zero_padding (ZeroPa  (None, 256, 2, 134)      0         
 dding2D)                                                        

 second_convolution_layer (C  (None, 80, 1, 132)       122960    
 onv2D)                                                          

 second_dropout (Dropout)    (None, 80, 1, 132)        0         

 flatten (Flatten)           (None, 10560)             0         

 first_dense_layer (Dense)   (None, 256)               2703616   

 third_dropout (Dropout)     (None, 256)               0         

 second_dense_layer (Dense)  (None, 11)                2827      

 softmax_function (Activatio  (None, 11)               0         
 n)                                                              

 second_reshape (Reshape)    (None, 11)                0         

=================================================================
Total params: 2,830,427
Trainable params: 2,830,427
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code 0


"""

# =============================================== Train Model ======================================================= #
"""
In classification, it's a little more complicated, but very similar. Predicted classes are based on probability. 
The loss is therefore also based on probability. In classification, the neural network minimizes the likelihood to 
assign a low probability to the actual class. The loss is typically categorical_crossentropy.

loss and val_loss differ because the former is applied to the train set, and the latter the test set. As such, 
the latter is a good indication of how the model performs on unseen data. You can get a validation set by using 
validation_data=[x_test, y_test] 
It's best to rely on the val_loss to prevent overfitting. Overfitting is when the model fits the training data too closely, 
and the loss keeps decreasing while the val_loss is stale, or increases.


The model will take the 2, 128 arrays from the 1000 bunch of these and assign it to a modulation type. That is the 
label list will have 220000 labels from 0 to 10 saying which modulation type it is
"""
no_epoch = 100
batch_size = 1024



history = model.fit(x_Train, label_Train, batch_size = batch_size, epochs = no_epoch, verbose = 1,
                    validation_data = (x_Test, label_Test), callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 0, mode = 'auto',
                                         baseline = None, restore_best_weights = False)])

model.save_weights("saved_model_weights/bytes_version")
model.save("complete_saved_model/bytes_version")

score = model.evaluate(x_Test, label_Test, verbose = 1, batch_size = batch_size)
print(score)
