# Detection-of-Adversarial-Attacks-on-Machine-Learning-Based-Wireless-Communication-Systems
Repository of the code for my university research paper 2022 - 2023 (attached in the repository - I forget if I signed something that says I can't show my paper, oh well).

Note - The objective of this code was not to display the optimisation of such a python project, but to produce results whilst I was in the depths of my studies.

This is for future reference and memory.  

Paper influenced by and adversarial attacks produced by Meysam Sadeghi - https://github.com/meysamsadeghi/Security-and-Robustness-of-Deep-Learning-in-Wireless-Communication-Systems/tree/master/Adv_Attack_Modulation_Classification

These were run on pycharm with all the required libraries downloaded - Tensorflow 2.0, Keras, Numpy etc.

Relies upon several files:
1) The 2016 synthetic radio signal dataset produced by DeepSig - https://www.deepsig.ai/datasets
2) The output of the model.py file to use in subsequent detection methods.
3) The reliance of the "valid indexes" for different SNR values in the white box detection, and black box detection.
4) And probably several more, but each file has comments describing the process and the paper is there to help.


The detection method primarily focuses on the softmax values as there are exploitations learned from this layer.

White Box Attack -
This attack in general is very computationally expensive, and is very prone to crashing. Hence, only some SNR values are used for experimentation. The main idea is to run each data point through the attack and output the softmax values and then conduct statistical analysis upon it by using the metrics and classifiers file.

The exhausting part of this process is that the attack itself is time consuming, 2 hours to complete 500 data points, when there are roughly 1000 data points per modulation type per SNR value. 
The ideal amount of data input's adversarially attacked without crashing for myself was 500 data points. This will vary for different computers, however, I ran this on my laptop and computer simultaneously, and my computer has better hardware so who knows. 500 data points might just be the optimal amount. 


Black Box Attack - 
Much faster than the white box attack, should see results in an hour or so, maybe less. 

Produces UAPs (Universal Adversarial Perturbation) - these are based upon  data points randomly chosen and then obtaining their gradient direction. Singular value decomposition is applied to find the first principal direction. Pretty interesting. ICM, CCM and MIX are the groups the data points are chosen to create the UAP from.

Further information is on the paper itself. 

Hopefully I'll remember the process of how I did this in the years to come, hopefully.






