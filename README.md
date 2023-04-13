# DEEP PROBABILISTIC MODEL FOR LOSSLESS SCALABLE POINT CLOUD ATTRIBUTE COMPRESSION
* **Authors**:
[Dat T. Nguyen](https://scholar.google.com/citations?hl=en&user=uqqqlGgAAAAJ),
[Kamal Gopikrishnan Nambiar](https://www.lms.tf.fau.de/person/nambiar-kamal-gopikrishnan/)
[André Kaup](https://scholar.google.de/citations?user=0En1UwQAAAAJ&hl=de),

* **Affiliation**: Friedrich-Alexander University Erlangen-Nuremberg, 91058 Erlangen, Germany

* **Accepted to**: [[IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)]](https://ieeexplore.ieee.org/xpl/conhome/1000002/all-proceedings)

* **Links**: [[Arxiv]](https://arxiv.org/abs/2303.06517)

## Description

- Abstract: In this work, we build an end-to-end multiscale point cloud attribute coding method (MNeT) that progressively projects the attributes onto multiscale latent spaces. The multiscale architecture provides an accurate context for the attribute probability modeling and thus minimizes the coding bitrate with a single network prediction. Besides, our method allows scalable coding that lower quality versions can be easily extracted from the losslessly compressed bitstream. We validate our method on a set of point clouds from MVUB and MPEG and show that our method outperforms recently proposed methods and on par with the latest G-PCC version 14. Besides, our coding time is substantially faster than G-PCC. 

- This is a Pytorch implementation of the scalable lossless attribute compression from the MNeT paper.

## Requirments: packages in environment.yml

- pytorch 1.9.0, py3.9_cuda11.1_cudnn8.0.5_0 
- MinkowskiEngine 0.5
- [torchac](https://github.com/fab-jul/torchac)

## Getting started
- Install the dependencies using the conda environment.yml file
```shell
conda env create --name mink --file environment.yml
```
## Dataset
- We use four datasets with color attributes: MPEG Owlii, MPEG 8i, MVUB, and MPEG CAT1 (except point clouds used for testing). All points clouds are partitioned into a set of point with maximum bounding box of 64. 
- Example: the following command to perform the partitioning for Owlii dataset: 
```
    python3 -m DataPreprocessing.ds_pc_octree_blocks datasets/Owlii datasets/Owlii_bl64 --vg_size 1024 --level 4 
```
The partitioned blocks are written into the datasets/Owlii_bl64 directory, doing the same for other datasets
- Randomly split blocks to training and validating set: 
``` 
    python3 -m DataPreprocessing.ds_select_randomly_move datasets/ datasets/ 0.3
``` 
- An example training set is located in ExampleTrainingSet/
## Training

    python3 -m Training.train_MNeT -trainset ExampleTrainingSet/TrainSet/  -validset ExampleTrainingSet/ValidSet  -flag train_2510 -outputmodel Model/  -nores 8  -useDA 0    --quantize   -lr 50 -noscale 3  -noltfil 5 -nobins 16  -ngpus 1  -batch 2   -bacc 1

## Encoding

    python3 -m Encoder.MNeT_Encoder -level 10 -ply  ../Datasets/TestPCs/10bits/ricardo_0010.ply -output Output/ -model Model/best_val_checkpoint_model__lr_50_b_32_da_0_nores_8_schedule_50.75_nobins_26_noltfil_5-epoch=125-val_loss=10.42.ckpt -signaling 2510_pooling -nobins 26 -nofil 5 -noscale 3

Checkpoints can be downloaded from [here](https://faubox.rrze.uni-erlangen.de/getlink/fiJLJVHMvF5dRsGoNgWkJt/)



