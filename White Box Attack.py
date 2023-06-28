import numpy as np
import tensorflow as tf
import pickle

# ====================================================== Loading Data ================================================== #
dataset = pickle.load(open("RML2016.10a_dict.pkl", "rb"),
                      encoding = "bytes")  # encoding = "latin1" or encoding = "bytes", sources say bytes are better https://stackoverflow.com/a/47882939
snr_S, mod_S = map(lambda j: sorted(list(set(map(lambda x: x[j], dataset.keys())))), [1, 0])

dataset_Values = []
dataset_Keys = []

for mod in mod_S:
    for snr in snr_S:
        dataset_Values.append(dataset[(mod, snr)])
        for i in range(len(dataset[(mod, snr)])):
            dataset_Keys.append((mod, snr))

x_Input = np.vstack(dataset_Values)

no_Samples = 220000
# for i in range(100):
#     print(dataset_Values[0][0][1][i])


label_S = tf.keras.utils.to_categorical(list(map(lambda x: mod_S.index(dataset_Keys[x][0]), range(no_Samples))))

loss_object = tf.keras.losses.CategoricalCrossentropy()
# ====================================================== Importing Trained Model ================================================== #
pretrained_model = tf.keras.models.load_model("complete_saved_model/bytes_version")


PNR = 10
SNR = 0

file_Name = f"{SNR}SNR Valid Indexes.txt"

PNR = 10**(PNR/10)
SNR = 10**(SNR/10)

# ============================================================== ATTACK ============================================================ #

def create_Adverse_Perturb(input_Image, input_Label):
    with tf.GradientTape() as tape:
        tape.watch(input_Image)
        prediction = pretrained_model(input_Image)
        loss = loss_object(input_Label, prediction)
        gradient = tape.gradient(loss, input_Image)
        signed_grad = tf.sign(gradient)
    return signed_grad

def adversarial_Example(input_Image, input_Label, no_Class):
    input_Image = tf.convert_to_tensor(input_Image)
    input_Label = tf.convert_to_tensor(input_Label)
    input_Label = tf.one_hot(np.argmax(input_Label), 11)
    input_Label = tf.reshape(input_Label, (1, 11))

    eps_Acc = 0.00001 * np.linalg.norm(input_Image)
    eps_Vector = np.zeros([no_Class])

    for cls in range(no_Class):
        test_Label = tf.cast(tf.convert_to_tensor(np.eye(no_Class)[cls, :]), dtype = tf.float32)
        test_Label = tf.reshape(test_Label, (1, no_Class))

        signed_Grad = create_Adverse_Perturb(input_Image, test_Label)
        norm_Adv_Per = signed_Grad / (np.linalg.norm(signed_Grad) + 0.000000000001)
        eps_Max = np.sqrt(PNR/SNR)
        eps_Min = 0
        no_Iter = 0
        while (eps_Max - eps_Min > eps_Acc) and (no_Iter < 30):
            no_Iter += 1
            epsilon = (eps_Max + eps_Min) / 2
            perturb_Added = input_Image + (epsilon * norm_Adv_Per)

            pred_Prob = pretrained_model.predict(perturb_Added, verbose = 0)

            compare = np.equal(np.argmax(pred_Prob), np.argmax(input_Label))
            if compare:
                eps_Min = epsilon
            else:
                eps_Max = epsilon

            eps_Vector[cls] = epsilon + eps_Acc

    false_Class = np.argmin(eps_Vector)
    min_Epsilon = np.min(eps_Vector)
    worst_Label = tf.cast(tf.convert_to_tensor(np.eye(no_Class)[false_Class, :]), dtype = tf.float32)
    worst_Label = tf.reshape(worst_Label, (1, no_Class))
    signed_Grad = create_Adverse_Perturb(input_Image, worst_Label)
    norm_Adv_Per = signed_Grad / (np.linalg.norm(signed_Grad) + 0.000000000001)
    adv_Perturb = min_Epsilon * norm_Adv_Per
    adv_Image = input_Image + adv_Perturb

    return adv_Image, adv_Perturb, false_Class, min_Epsilon,


# ====================================================== Generating the softmax outputs ======================================== #

with open(file_Name) as file:
    data_Index_Match = [int(i) for i in file]
print("Length of data index match: ", len(data_Index_Match))

"""
This is where it becomes very time inducing to obtain results. On my computer for 500 data inputs it takes two hours or so.
As I said in the report, the white-box attack only creates perturbations for the data inputs that the neural network can correctly classify.
Therefore, the data index match is a file filled with those indexes for a specific SNR level.


"""

""" The offset would need to increased 500 every time this file has finished.
For example, for 0 SNR there are 8000 or so data index matches. So if every time the file finishes 500 data indexes have been done, then you will have to increment manually to obtain the further 7500 more.

In short, increase by 500 every time.

After the softmax outputs have been outputted, its best to copy and paste them into an excel file, and then restart the process.

As this is a very big computation occurring, it is hard to optimise as it will crash if more number of values are being processed and progress will be lost.
"""
offset = 0
num = 500

softmax_Output_Normal = []
softmax_Output_Adverse = []

for i in range(num):
    print(i)

    value_Index = data_Index_Match[offset + i]
    adv_Image, _, _, _ = adversarial_Example(x_Input[[value_Index]], label_S[value_Index], 11)
    softmax_Output_Normal.append(pretrained_model.predict(x_Input[[value_Index]], verbose = 0))
    softmax_Output_Adverse.append(pretrained_model.predict(adv_Image, verbose = 0))

file_Name_Normal = f"{PNR}PNR {SNR}SNR Normal Output.txt"

f = open(file_Name_Normal, "w")
for i in range(num):
    for n in range(11):
        f.write(f"{softmax_Output_Normal[i][0][n]}\n")
        print(softmax_Output_Normal[i][0][n])
    print(" ")


file_Name_Adverse = f"{PNR}PNR {SNR}SNR Adverse Output.txt"
f = open(file_Name_Adverse, "w")
for i in range(num):
    for n in range(11):
        f.write(f"{softmax_Output_Adverse[i][0][n]}\n")
        print(softmax_Output_Adverse[i][0][n])
    print(" ")
