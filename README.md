# CS5824_Team6
This is the GitHub repository for the CS5824 19Fall course project (Team6). The project is aimed to 1) reproduce the results in 
the paper titled "Physics-Informed Generative Adversarial Networks by Incorporating Conservation Laws" [1], 2) evaluate the 
findings by running the code and comparing the results we get to the results in Ref [1], 3) get ourselves familiar with the 
principles of how GANs or specifically physics-informed GANs work.

[1] Zeng, Yang, Jinlong Wu, and Heng Xiao. "Physics-Informed Generative Adversarial Networks by Incorporating Conservation Laws." Bulletin of the American Physical Society 63 (2018).



# How to run the code
You need to have a GPU to run the code to save time as this code can be slow on the GPU. If you are a VT student, you may need to apply for an allocation on the ARC system including NewRiver, CasCades and HuckleBerry. If you have allocation there already, then please follow the following steps.

# Step 1
Login to a cluster first. For example, if you want to acesss NewRiver, try

ssh -X your_PID@newriver1.arc.vt.edu

You need to replace "your_PID" with your PID. For me, I use "ssh -X weich97@newriver1.arc.vt.edu"

After you do this, VT requires your password to your account and you also need to push your DUO to allow the access.

For the Huckleberry, try

ssh -X your_PID@huckleberry1.arc.vt.edu

# Step 2
Then you need to apply a node with a certain number of CPU cores and GPUs for a period of time. On NewRiver, try

qsub -I -lnodes=1:ppn=2:gpus=2 -q p100_normal_q -l walltime=04:00:00 -W group_list=newriver -A your_allocation_name

This sentence means that a node with 2 CPU cores and 2 GPUs is applied for 4 hours. You need to replace "your_allocation_name" with the allocation name you have and make sure you are added into that allocation. Our group's allocation name is "vt_aoe_dl", i.e.,

qsub -I -lnodes=1:ppn=2:gpus=2 -q p100_normal_q -l walltime=04:00:00 -W group_list=newriver -A vt_aoe_dl

If you want to access HuckleBerry, then try

salloc --time=04:00:00 -N 1 -n 8 --gres=gpu:2 --partition=normal_q --account=vt_aoe_dl

# Step 3
Then you need to load some modules and activate the Anaconda environment on the nodes you applied. On NewRiver, try

'''
module load Anaconda/5.1.0
module load cuda/9.0.176
module load cudnn/7.1
'''

If you are using Huckleberry, then try

module load gcc cuda Anaconda3 jdk
source activate powerai16_ibm

# Step 4
Finally you can run the code. cd to the directory where you store the code, and try

python Serial_PIGANs_PF_dataset.py

You may need to read the code first, as this code requires that you prepared the data, created a directory to store the results, e.t.c

# To generate the training data
Just run using "python write_tfrecord.py". You can change the mesh size, but you should also change the network size and the convolution kernel size and the strides in the "Serial_PIGANs_PF_dataset.py" (you do not need to do so for 32x32 to 512x512 grids). This file also has a normalization so that the train data can be normalized.

# Some important notes
The source code can be found at git@github.com:zengyang7/Parallel-PIGANs.git. The source code may be used but requires citing:

[1] Zeng, Yang, Jinlong Wu, and Heng Xiao. "Physics-Informed Generative Adversarial Networks by Incorporating Conservation Laws." Bulletin of the American Physical Society 63 (2018).


If there is any question about running the code, please email weich@vt.edu
