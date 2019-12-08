# CS5824_Team6
This is the GitHub repository for the CS5824 19Fall course project (Team6). The project is aimed to 1) reproduce the results in 
the paper titled "Physics-Informed Generative Adversarial Networks by Incorporating Conservation Laws" [1], 2) evaluate the 
findings by running the code and comparing the results we get to the results in Ref [1], 3) get ourselves familiar with the 
principles of how GANs or specifically physics-informed GANs work.

[1] Zeng, Yang, Jinlong Wu, and Heng Xiao. "Physics-Informed Generative Adversarial Networks by Incorporating Conservation Laws." Bulletin of the American Physical Society 63 (2018).

[2] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

# What have been reproduced in Ref [1]?
The paper can be summarized into 4 parts. the first part deals with the introduction and evolution of the GANs models. General GANs are introduced first, following by specific GANs and physcis-informed GANs. The main aim of the paper is to apply the physics-informed GANs model to simulate a simple potential flow (although this paper has a simpler test case to prove the GANs is working properly).

# Background
Traditional Computational Fluid Dynamics (CFD) methods are commonly computational expensive in solving a system of Partial Differential Equations (PDEs) on well refined meshes, especially for turbulent and high Reynolds number flows. To generate the numerical solutions, we usually need to iteratively solve a system of nonlinear partial differential equations. Iteratively solving these equations on well refined meshes can be computationally expensive, therefore researchers in the CFD area have been thinking about new ways to generate numerical solutions with high accuracy faster. Here some figures are shown using CFD to generate numerical solutions for some applications. Fortunately, the development of machine learning (ML) or deep learning (DL) technique provides us with a completely innovated way to generate solutions for some fluid dynamics problems in hours, or even in minutes. This is a multi-disciplinary field which has aroused a lot of interestes from the areas of Aerospace Engineering and Data Sciences. As Aerospace Engineers, the ultimate goal is to solve CFD problems numerically using ML techniques such as physics-informed GANs, as they are computationally cheaper than the traditional CFD methods.

![image](Screenshots/LDC.png)
![image](Screenshots/2Dstep.png)

Having the capability of generating false data, the generative adversarial networks (GANs) have been regarded as one of the most promising deep learning methods. GANs is composed of two neural networks, one of which is generative neural network and the other is discriminative neural network so that the two neural networks can compete with each other to generate some false data which mimic the true data. When applying GANs to physical problems, there may be some serious issues. The first issue is that the generated data may not satisfy physical conservation laws or constraints due to its poor ability to extract complex physical features correctly. The second issue is that the training process may become more difficult and time-consuming after importing some physical constraints to the model.

Zeng's paper [1] is about using a physics-informed GANs model (PI-GANs) to simulate a family of potential flows (uniform flow + source flow). In PI-GANs, physical information such as the mass conservation law (for an incompressible flow) is integrated to GANs as a penalty term. This penalty term embedded to the generator to enforce the generated data inform the physical information. Although the flow itself studied in this project is elementary, the goal of this project is to see whether this PI-GANs can generate some “true” flows that mimic the real flows satisfying physical constraints, and it is a good start for future training on more complicated flows.

# Theory
Goodfellow et al. [2] firstly proposed GANs in 2014. The objective function of GANs used in their work was given as:
![image](Screenshots/GANs_formula.png)

where G and D are generator and discriminator, respectively, z the latent variables which are sampled from a given distribution p<sub>z</sub>(z) such as a uniform and Gaussian distribution, X the given training samples, p<sub>data</sub>(X) the distribution of these training samples.

Physical constraints can be usually denoted as H(X)<=0. To evaluate whether the generated data of GANS satisfies some physical constraints, the constaint term is included in a loss function term given as:
![image](Screenshots/Constraints.png)

This term is integrated into the loss function of GANS through:
![image](Screenshots/Loss.png)

where $\lambda$ is a tunning factor. 

# How to run the code
You need to have a GPU to run the code to save time as this code can be slow on the GPU. If you are a VT student, you may need to apply for an allocation on the ARC system including NewRiver, CasCades and HuckleBerry. If you have allocation there already, then please follow the following steps.

## Step 1
Login to a cluster first. For example, if you want to acesss NewRiver, try

ssh -X your_PID@newriver1.arc.vt.edu

You need to replace "your_PID" with your PID. For me, I use "ssh -X weich97@newriver1.arc.vt.edu"

After you do this, VT requires your password to your account and you also need to push your DUO to allow the access.

For the Huckleberry, try

ssh -X your_PID@huckleberry1.arc.vt.edu

## Step 2
Then you need to apply a node with a certain number of CPU cores and GPUs for a period of time. On NewRiver, try

qsub -I -lnodes=1:ppn=2:gpus=2 -q p100_normal_q -l walltime=04:00:00 -W group_list=newriver -A your_allocation_name

This sentence means that a node with 2 CPU cores and 2 GPUs is applied for 4 hours. You need to replace "your_allocation_name" with the allocation name you have and make sure you are added into that allocation. Our group's allocation name is "vt_aoe_dl", i.e.,

qsub -I -lnodes=1:ppn=2:gpus=2 -q p100_normal_q -l walltime=04:00:00 -W group_list=newriver -A vt_aoe_dl

If you want to access HuckleBerry, then try

salloc --time=04:00:00 -N 1 -n 8 --gres=gpu:2 --partition=normal_q --account=vt_aoe_dl

## Step 3
Then you need to load some modules and activate the Anaconda environment on the nodes you applied. On NewRiver, try

module load Anaconda/5.1.0

module load cuda/9.0.176

module load cudnn/7.1


If you are using Huckleberry, then try

module load gcc cuda Anaconda3 jdk

source activate powerai16_ibm

## Step 4
Finally you can run the code. cd to the directory where you store the code, and try

python Serial_PIGANs_PF_dataset.py

You may need to read the code first, as this code requires that you prepared the data, created a directory to store the results, e.t.c

# To generate the training data
Just run using "python write_tfrecord.py". You can change the mesh size, but you should also change the network size and the convolution kernel size and the strides in the "Serial_PIGANs_PF_dataset.py" (you do not need to do so for 32x32 to 512x512 grids). This file also has a normalization so that the train data can be normalized.

# Some important notes
The original source code can be found at git@github.com:zengyang7/Parallel-PIGANs.git. The source code may be used but requires citing:

[1] Zeng, Yang, Jinlong Wu, and Heng Xiao. "Physics-Informed Generative Adversarial Networks by Incorporating Conservation Laws." Bulletin of the American Physical Society 63 (2018).


The code on this project is modified based on the code in Ref [1]. If there is any question about running the code in this repository, please email weich@vt.edu. Thanks!
