# SeaNet
This project provides the code and results for 'Lightweight Salient Object Detection in Optical Remote-Sensing Images via Semantic Matching and Edge Alignment', IEEE TGRS, 2023. [IEEE link](https://ieeexplore.ieee.org/document/10015064) and [arxiv link](https://arxiv.org/abs/2301.02778) [Homepage](https://mathlee.github.io/)

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/SeaNet/blob/main/image/SeaNet.png">
   </div>
   
   
# Requirements
   python 3.7 + pytorch 1.9.0


# Saliency maps
   We provide saliency maps of our SeaNet on ORSSD, EORSSD, and additional ORSI-4199 datasets in './models/saliency_maps.zip'.
      
   ![Image](https://github.com/MathLee/SeaNet/blob/main/image/table.png)
   
   
# Training
   We use data_aug.m for data augmentation. 
   
   Modify paths of datasets, then run train_SeaNet.py.

Note: our main model is under './model/SeaNet_models.py'



# Pre-trained model and testing
1. We provide the pre-trained models in './models/'.

2. Modify paths of pre-trained models and datasets.

3. Run test_SeaNet.py.

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2023_SeaNet,
                author = {Gongyang Li and Zhi Liu and Xinpeng Zhang and Weisi Lin},
                title = {Lightweight Salient Object Detection in Optical Remote-Sensing Images via Semantic Matching and Edge Alignment},
                journal = {IEEE Transactions on Geoscience and Remote Sensing},
                volume = {61},
                year = {2023},
                doi = {10.1109/TGRS.2023.3235717},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
