import numpy as np
import pickle
import tensorflow as tf
import random

## tensorflow 2.10

# ================================================== Loading Data ==================================================== #
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

# ================================================ Specify SNR and PSR =============================================== #

SNR = 10
PSR = 0
PNR = PSR + SNR
""" README: THIS PARTICULAR FILE CREATES A UAP BASED ON ONLY CCM INPUTS AS MENTIONED IN THE REPORT
THIS SECTION FIRST IMPORTS THE INDEXES WHERE THE NEURAL NETWORK CAN CORRECTLY CLASSIFY (CCM) 
AND THEN REMOVES THE TRAINING INDEXES USED TO TRAIN THE NEURAL NETWORK
OVERALL, YOU ACHIEVE INDEXES OF ONLY CCM INPUTS
"""

""" 
Loads the indexes that the NN can correctly identify
"""

file_Name_Valid_Indexes = f"{SNR}SNR Valid Indexes.txt"
with open(file_Name_Valid_Indexes) as file:
    data_Index_Match = [int(i) for i in file]
""" 
Loads the indexes that the NN trained on
"""
file_Name_Train_Index = "Train Indexes.txt"
with open(file_Name_Train_Index) as file:
    NN_Train_Indexes = [int(i) for i in file]
""" 
Removes the indexes that the NN trained on
"""
for num in data_Index_Match[:]:
    if num in NN_Train_Indexes:
        data_Index_Match.remove(num)

""" This produces only CCM used to create the UAP """


length = len(data_Index_Match)
print("The length is", length)
# ======================================= Partitions Data in to training and testing ======================================#

"""
Partitions the data into training for the UAP and testing if the UAP is accurate
"""
np.random.seed(2022)
train_Index_File = np.random.choice(range(0, len(data_Index_Match)), size = int(len(data_Index_Match) / 2),
                                    replace = False)
test_Index_File = list(set(range(0, int(len(data_Index_Match) / 2))) - set(train_Index_File))
train_Index = []
test_Index = []

for i in range(len(train_Index_File)):
    train_Index.append(data_Index_Match[train_Index_File[i]])

for i in range(len(test_Index_File)):
    test_Index.append(data_Index_Match[test_Index_File[i]])
#
train_X = x_Input[train_Index]
test_X = x_Input[test_Index]

train_Label = tf.keras.utils.to_categorical(list(map(lambda x: mod_S.index(dataset_Keys[x][0]), train_Index)))
test_Label = tf.keras.utils.to_categorical(list(map(lambda x: mod_S.index(dataset_Keys[x][0]), test_Index)))

# ======================================= Loads the saved model =================================================#
pretrained_model = tf.keras.models.load_model("complete_saved_model/bytes_version")
loss_object = tf.keras.losses.CategoricalCrossentropy()

# ======================================= Starts Algorithm 2 ====================================================#


# Finds average norm of all inputs and squares to obtain power.
signal_Power = (np.sum(np.linalg.norm(train_X.reshape([-1, 256]), axis = 1)) / train_X.shape[0]) ** 2

# Obtains PSR in linear form
psr_Aid = 10 ** ((PNR - SNR) / 10)

# Assumes that S * P/S instead of S + N * P/S and returns perturbation power norm
epsilon_Uni = np.sqrt(signal_Power * psr_Aid)


# vec_Grad_Un = np.zeros([num_Times, 1])

def create_Adverse_Perturb(input_Image_P, input_Label_P):
    """ Obtains the gradient """

    with tf.GradientTape() as tape:
        tape.watch(input_Image_P)
        prediction = pretrained_model(input_Image_P)
        loss = loss_object(input_Label_P, prediction)
        gradient = tape.gradient(loss, input_Image_P)
        signed_grad = tf.sign(gradient)
    return signed_grad


acc_Optimal_Grad = 1
N_n = 50
num_Times = 5
vec_Grad_N = np.zeros([num_Times, 1])

for rnd in range(num_Times):  # this is the number of times we randomly select N data points
    print("rnd: ", rnd, " Out of: ", num_Times)
    # Randomly selects inputs
    np.random.seed()
    select_Inputs = np.random.choice(range(train_X.shape[0]), size = N_n, replace = False)

    # grad_matrix_un = np.zeros([N_n, 256])
    grad_matrix_n = np.zeros([N_n, 256])

    for ctr_index in range(N_n):
        print("ctr_index: ", ctr_index, "Out of: ", N_n)
        input_image = train_X[select_Inputs[ctr_index]].reshape([-1, 2, 128, 1])
        temp = np.asarray(create_Adverse_Perturb(tf.convert_to_tensor(input_image), tf.convert_to_tensor(
            train_Label[select_Inputs[ctr_index]].reshape(1, 11)))).reshape(1, 256)
        # grad_matrix_un[ctr_index, :] = temp
        grad_matrix_n[ctr_index, :] = temp / (np.linalg.norm(temp) + 0.00000001)

    _, _, v_n_T = np.linalg.svd(grad_matrix_n)
    grad_per_n = epsilon_Uni * (1 / np.linalg.norm(v_n_T.T[:, 0])) * v_n_T.T[:, 0]

    # _, _, v_un_T = np.linalg.svd(grad_matrix_un)
    # grad_per_un = epsilon_Uni * (1 / np.linalg.norm(v_un_T.T[:, 0])) * v_un_T.T[:, 0]

    num_samples = test_X.shape[0]
    accuracy_Grad_UN = 0
    accuracy_Grad_N = 0

    for i_N in range(num_samples):
        print("i_N: ", i_N, "Out of: ", num_samples)
        input_image = test_X[i_N].reshape([-1, 2, 128, 1])

        # pred_grad_un = np.argmax(pretrained_model.predict(input_image + grad_per_un.reshape([1, 2, 128, 1]), verbose = 0))
        pred_grad_n = np.argmax(pretrained_model.predict(input_image + grad_per_n.reshape([1, 2, 128, 1]), verbose = 0))

        # if np.argmax(test_Label[i_N]) == pred_grad_un:
        # accuracy_Grad_UN = accuracy_Grad_UN + 1
        if np.argmax(test_Label[i_N]) == pred_grad_n:
            accuracy_Grad_N = accuracy_Grad_N + 1

    # acc_Gr_Un = accuracy_Grad_UN / num_samples
    acc_Gr_N = accuracy_Grad_N / num_samples

    # vec_Grad_Un[rnd] = acc_Gr_Un
    vec_Grad_N[rnd] = acc_Gr_N
    if acc_Optimal_Grad > acc_Gr_N:
        acc_Optimal_Grad = acc_Gr_N
        optimal_Grad = grad_per_n.reshape([256])

print(*optimal_Grad, sep = '\n')
acc_Grad_N = np.sum(vec_Grad_N) / num_Times
print('grad_n acc', acc_Grad_N)

# ============================================== Stores UAP =========================================================== #


file_Name_UAP = f"UAP_SNR{SNR}_PSR{PSR}_PNR{PNR}.txt"
f = open(file_Name_UAP, "w")
for i in range(len(optimal_Grad)):
    f.write(f"{optimal_Grad[i]}\n")

"""  Stores UAP in attack matrix """
attack = []
for i in range(len(optimal_Grad)):
    attack.append(optimal_Grad[i])

attack = np.array(attack)

# ================================================ Applying UAP and Outputting Softmax Values =================================================== #
softmax_Output_Attack = []
softmax_Output_Normal = []

"""##################Important################

In the report, we create a UAP based on CCM, ICM and MIX data inputs. This SPECIFIC file creates a UAP based on CCM. 
There are two more files.

NOW to APPLY (NOT CREATE) this to different groups of data inputs.

1. To apply to only CCM - UNCOMMENT THE FIRST SECTION
2. To apply to only ICM - UNCOMMENT THE SECOND SECTION
3. To apply to MIX - UNCOMMENT THE THIRD SECTION

WHEN TESTING A SPECIFIC GROUP YOU MUST COMMENT THE OTHER GROUPS OUT

THE SECTIONS ARE CLEARLY LABELLED. 
"""
# # ================================================ 1. APPLYING CCM UAP TO CCM =================================================== #
# """1. PLEASE UNCOMMENT THIS SECTION TO PRODUCE THE SOFTMAX OUTPUT FOR CCM DATA INPUTS THAT ARE AFFECTED BY THE CCM BASED UAP """
# length = len(data_Index_Match)
# for i in range(length):
#     print(i)
#     value_Index = data_Index_Match[i]
#     input_image = np.array(x_Input[[value_Index]]).reshape([-1, 2, 128, 1])
#     softmax_Output_Normal.append(pretrained_model.predict(input_image, verbose = 0))
#     softmax_Output_Attack.append(pretrained_model.predict(input_image + attack.reshape([1, 2, 128, 1]), verbose = 0))
#
# """ Stores in text files which can be used immediately in the metrics and classifiers"""
#
# file_Name_Normal = f"{PSR}PSR {SNR}SNR CCM CCM Normal.txt"
#
# f = open(file_Name_Normal, "w")
# for i in range(length):
#     for n in range(11):
#         f.write(f"{softmax_Output_Normal[i][0][n]}\n")
#         print(softmax_Output_Normal[i][0][n])
#
# file_Name_Adverse = f"{PSR}PSR {SNR}SNR CCM CCM Adverse.txt"
# f = open(file_Name_Adverse, "w")
# for i in range(length):
#     for n in range(11):
#         f.write(f"{softmax_Output_Attack[i][0][n]}\n")
#         print(softmax_Output_Attack[i][0][n])

# # ================================================ 1. END =================================================== #


# ================================================ 2. APPLYING CCM UAP to ICM DATA INPUTS =========================================#
# """2. PLEASE UNCOMMENT THIS SECTION TO PRODUCE THE SOFTMAX OUTPUT FOR ICM DATA INPUTS THAT ARE AFFECTED BY THE CCM BASED UAP """
# """
# Within the data, indexes from 15000 to 16000 are for SNR  = 10dB, and this increments every 20000. The below creates a list of integers from 15000 to 16000, etc.
# """
# ranges = [(15000, 16000), (35000, 36000), (55000, 56000), (75000, 76000),
#           (95000, 96000), (115000, 116000), (135000, 136000), (155000, 156000),
#           (175000, 176000), (195000, 196000), (215000, 216000)]
#
# all_numbers = []
# for r in ranges:
#     all_numbers.extend(range(r[0], r[1] + 1))
#
# """
# Removes the indexes that the NN trained on
# """
# for num in all_numbers[:]:
#     if num in NN_Train_Indexes:
#         all_numbers.remove(num)
#
# """
# Removes the indexes for CCM
# """
# for num in all_numbers[:]:
#     if num in data_Index_Match:
#         all_numbers.remove(num)
#
# """
# Gets 4000 different random numbers from the list. This is to limit it to 4000 ICM data inputs to apply the UAP to, if not
# it would have to apply up 10000 values, which takes a while
#
# """
# random_numbers = []
# while len(random_numbers) < 4000:
#     num = random.sample(all_numbers, 1)[0]
#     random_numbers.append(num)
#
#
# length = len(random_numbers)
# for i in range(length):
#     print(i)
#     value_Index = random_numbers[i]
#     input_image = np.array(x_Input[[value_Index]]).reshape([-1, 2, 128, 1])
#     softmax_Output_Normal.append(pretrained_model.predict(input_image, verbose = 0))
#     softmax_Output_Attack.append(pretrained_model.predict(input_image + attack.reshape([1, 2, 128, 1]), verbose = 0))
# """ Stores in text files which can be used immediately in the metrics and classifiers"""
#
# file_Name_Normal = f"{PSR}PSR {SNR}SNR CCM ICM Normal.txt"
#
# f = open(file_Name_Normal, "w")
# for i in range(length):
#     for n in range(11):
#         f.write(f"{softmax_Output_Normal[i][0][n]}\n")
#         print(softmax_Output_Normal[i][0][n])
#
# file_Name_Adverse = f"{PSR}PSR {SNR}SNR CCM ICM Adverse.txt"
# f = open(file_Name_Adverse, "w")
# for i in range(length):
#     for n in range(11):
#         f.write(f"{softmax_Output_Attack[i][0][n]}\n")
#         print(softmax_Output_Attack[i][0][n])

# ================================================ 2. END =========================================#



# # =================================================3. APPLYING CCM UAP TO MIX =================================#
"""3. PLEASE UNCOMMENT THIS SECTION TO PRODUCE THE SOFTMAX OUTPUT FOR MIX DATA INPUTS THAT ARE AFFECTED BY THE CCM BASED UAP """

ranges = [(15000, 16000), (35000, 36000), (55000, 56000), (75000, 76000),
          (95000, 96000), (115000, 116000), (135000, 136000), (155000, 156000),
          (175000, 176000), (195000, 196000), (215000, 216000)]

all_numbers = []
for r in ranges:
    all_numbers.extend(range(r[0], r[1] + 1))
random_numbers = []

"""
Gets 4000 different random numbers from the list. This is to limit it to 4000 ICM data inputs to apply the UAP to, if not
it would have to apply up 10000 values, which takes a while

"""
while len(random_numbers) < 4000:
    num = random.sample(all_numbers, 1)[0]
    random_numbers.append(num)


length = len(random_numbers)
for i in range(length):
    print(i)
    value_Index = random_numbers[i]
    input_image = np.array(x_Input[[value_Index]]).reshape([-1, 2, 128, 1])
    softmax_Output_Normal.append(pretrained_model.predict(input_image, verbose = 0))
    softmax_Output_Attack.append(pretrained_model.predict(input_image + attack.reshape([1, 2, 128, 1]), verbose = 0))
""" Stores in text files which can be used immediately in the metrics and classifiers"""

file_Name_Normal = f"{PSR}PSR {SNR}SNR CCM MIX Normal.txt"

f = open(file_Name_Normal, "w")
for i in range(length):
    for n in range(11):
        f.write(f"{softmax_Output_Normal[i][0][n]}\n")
        print(softmax_Output_Normal[i][0][n])

file_Name_Adverse = f"{PSR}PSR {SNR}SNR CCM MIX Adverse.txt"
f = open(file_Name_Adverse, "w")
for i in range(length):
    for n in range(11):
        f.write(f"{softmax_Output_Attack[i][0][n]}\n")
        print(softmax_Output_Attack[i][0][n])
# # =================================================3. END =================================#
