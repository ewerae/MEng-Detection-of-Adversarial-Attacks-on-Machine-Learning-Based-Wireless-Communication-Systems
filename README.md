# Detection-of-Adversarial-Attacks-on-Machine-Learning-Based-Wireless-Communication-Systems
Repository of the code for my university research paper 2022 - 2023 (attached in the repository).

Note - This code is not optimised in any sort, the focus was to produce results for my chosen topic. 

Influenced and adversarial attacks produced by Meysam Sadeghi - https://github.com/meysamsadeghi/Security-and-Robustness-of-Deep-Learning-in-Wireless-Communication-Systems/tree/master/Adv_Attack_Modulation_Classification

These were run on pycharm with all the required libraries downloaded - Tensorflow 2.0, Keras, Numpy etc.

Relies upon several files:
1) The 2016 synthetic radio signal dataset produced by DeepSig - https://www.deepsig.ai/datasets
2) The output of the model.py file to use in subsequent detection methods.
3) The reliance of the "valid indexes" for different SNR values in the white box detection, and black box detection.
4) And probably several more, but each file has comments describing the process and the paper is there to help.


The detection method primarily focuses on the softmax values as there are exploitations learned from this layer/

White Box Attack -
The synthetic dataset has a range of SNR values, it has a lot of data points.
This attack in general is very computationally expensive, and is very prone to crashing. Hence, only some SNR values are used for experimentation. The main idea is to run each data point through the attack and output the softmax values and then conduct statistical analysis upon it by using the metrics and classifiers file.

The exhausting part of this process is that the attack itself is time consuming, 2 hours to complete 500 data points, when there are roughly 1000 data points per modulation type per SNR value.


Black Box Attack - 
Much faster than the white box attack, should see results in an hour or so. The meaning of ICM and CCM and MIX is clearer within the research paper. 





