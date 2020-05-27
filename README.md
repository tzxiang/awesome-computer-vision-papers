# awesome-computer-vision-papers

A list of papers and other resources on computer vision and deep learning. 



## Reviews

- A Gentle Introduction to Deep Learning for Graphs. [arXiv201912](https://arxiv.org/abs/1912.12693) [[Note]](https://zhuanlan.zhihu.com/p/106003590)
- A Comprehensive Survey on Graph Neural Networks. [arXiv201912](https://arxiv.org/abs/1901.00596) [[Note]](https://zhuanlan.zhihu.com/p/75307407)
- Research Guide: Model Distillation Techniques for Deep Learning, Derrick Mwiti, 2019.11 [[Blog]](https://heartbeat.fritz.ai/research-guide-model-distillation-techniques-for-deep-learning-4a100801c0eb)
- Graph Neural Networks: A Review of Methods and Applications, arXiv2019.7 [[Intro-Chinese]](https://mp.weixin.qq.com/s/wVc3w5U5HwG33uzHdIc-3w)
- A Review on Deep Learning in Medical Image Reconstruction, arXiv2019.6
- MNIST-C: A Robustness Benchmark for Computer Vision, arXiv2019.6 [[Code&Dataset]](https://github.com/google-research/mnist-c)
- Going Deep in Medical Image Analysis: Concepts, Methods, Challenges and Future Directions， arXiv2019.2
- Computer Vision for Autonomous Vehicles: Problems, Datasets and State-of-the-Art. [arXiv201704](https://arxiv.org/pdf/1704.05519v1.pdf) [[Resourses]](https://github.com/aleju/papers/blob/master/mixed/Computer_Vision_for_Autonomous_Vehicles_Overview.md)
- [2019TIV] A Survey of Autonomous Driving: Common Practices and Emerging Technologies
- [2014JMLR] Do we need hundreds of classifiers to solve real world classification problems



## SemanticSeg

> [awesome-semantic-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)
>
> [SemanticSegPaperCollection](https://github.com/shawnyuen/SemanticSegPaperCollection)
>
> [SegLoss](https://github.com/JunMa11/SegLoss): A collection of loss functions for medical image segmentation
>
> [Efficient-Segmentation-Networks](https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks)
>
> [U-Net and its variant code](https://github.com/LeeJunHyun/Image_Segmentation)
>
> 深度学习下的语义分割综述 [[Page]](https://mp.weixin.qq.com/s/MFUAloM_PEBmPRFfFPF-9g) [[Notes]](https://zhuanlan.zhihu.com/p/76418243)
>
> 三维语义分割概述及总结 [[Page]](https://mp.weixin.qq.com/s/3rK8gXuATm_v-X6DNBQNTw)
>
> Unpooling/unsampling deconvolution [[Note]](https://blog.csdn.net/qq_38410428/article/details/91363142)
>
> Some basic points: [align_corners](https://zhuanlan.zhihu.com/p/87572724)
>
> [Code: Semantic Segmentation Suite in TensorFlow](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)



**Review**

- Recent progress in semantic image segmentation, Artificial Intelligence Review, 2019
- Review of Deep Learning Algorithms for Image Semantic Segmentation, 2018 [[Blog]](https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57)

**arXiv**

- Divided We Stand: A Novel Residual Group Attention Mechanism for Medical Image Segmentation, [arXiv2019.12](https://arxiv.org/abs/1912.02079)
- Hard Pixels Mining: Learning Using Privileged Information for Semantic Segmentation, [arXiv2019.11](https://arxiv.org/abs/1906.11437)
- Hierarchical Attention Networks for Medical Image Segmentation, [arXiv2019.11](https://arxiv.org/abs/1911.08777) [eye line seg]
- Multi-scale guided attention for medical image segmentation, [arXiv2019.10](https://arxiv.org/abs/1906.02849) [[Code]](https://github.com/sinAshish/Multi-Scale-Attention)
- Adaptive Class Weight based Dual Focal Loss for Improved Semantic Segmentation, [arXiv2019.10](https://arxiv.org/abs/1909.11932)
- ELKPPNet: An Edge-aware Neural Network with Large Kernel Pyramid Pooling for Learning Discriminative Features in Semantic Segmentation, [arXiv2019.6](https://arxiv.org/abs/1906.11428)
- ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation, [arXiv2019.6](https://arxiv.org/abs/1906.09826) [[Code]](https://github.com/xiaoyufenfei/ESNet)
- FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation, [arXiv2019.3](https://arxiv.org/abs/1903.11816) [[Proj]](http://wuhuikai.me/FastFCNProject/) [[Code]](https://github.com/wuhuikai/FastFCN) [[Note]](https://mp.weixin.qq.com/s/1lMlSMS5xKc8k0QMAou45g) [JPU: Joint Pyramid Upsampling]
- ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation, [arXiv2016.6](https://arxiv.org/abs/1606.02147) [[Code]](https://github.com/TimoSaemann/ENet)

**Journal/Proceedings**

- [2019IJCV] **AdapNet++**: Self-Supervised Model Adaptation for Multimodal Semantic Segmentation [[Code]](https://github.com/DeepSceneSeg/AdapNet-pp)
- [2019NIPS] Zero-Shot Semantic Segmentation [[Code]](https://github.com/RohanDoshi2018/ZeroshotSemanticSegmentation)
- [2019NIPS] Grid Saliency for Context Explanations of Semantic Segmentation [[github]](https://github.com/boschresearch/GridSaliency-ToyDatasetGen)
- [2019NIPS] Region Mutual Information Loss for Semantic Segmentation
- [2019NIPS] Improving Semantic Segmentation via Dilated Affinity
- [2019NIPS] Correlation Maximized Structural Similarity Lossfor Semantic Segmentation
- [2019NIPS] Multi-source Domain Adaptation for Semantic Segmentation
- [2019ICCV] Boundary-Aware Feature Propagation for Scene Segmentation
- [2019ICCV] [Adaptive-sampling] Efficient Segmentation: Learning Downsampling Near Semantic Boundaries [[github]](https://github.com/dmitrii-marin/adaptive-sampling) (Reference: LIP: Local Importance-based Pooling, ICCV2019 [[github]](https://github.com/sebgao/LIP) [[Notes]](https://zhuanlan.zhihu.com/p/85841067))

- [2019ICCV] Selectivity or Invariance: Boundary-aware Salient Object Detection [[Proj&Code]](http://cvteam.net/projects/ICCV19-SOD/BANet.html)
- [2019ICCV] Recurrent U-Net for Resource-Constrained Segmentation
- [2019ICCV] Gated-SCNN: Gated Shape CNNs for Semantic Segmentation [[Code]](https://github.com/nv-tlabs/GSCNN) [[Proj]](https://nv-tlabs.github.io/GSCNN/)
- [2019ICCV] Visualizing the Invisible: Occluded Vehicle Segmentation and Recovery 
- [2019ICCV] ACE: Adapting to Changing Environments for Semantic Segmentation
- [2019ICCV] Asymmetric Non-local Neural Networks for Semantic Segmentation
- [2019ICCV] DADA: Depth-Aware Domain Adaptation in Semantic Segmentation
- [2019ICCV] ACFNet: Attentional Class Feature Network for Semantic Segmentation
- [2019ICCV] [EMANet] Expectation-Maximization Attention Networks for Semantic Segmentation [[github]](https://github.com/XiaLiPKU/EMANet)
- [2019ICCV] CCNet : Criss-Cross Attention for Semantic Segmentation [[github]](https://github.com/speedinghzl/CCNet)
- [2019ICCV] Gated-SCNN: Gated Shape CNNs for Semantic Segmentation

- [2019CVPR] **ESPNetv2**: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network [[Code]](https://github.com/sacmehta/ESPNetv2)
- [2019CVPR] Not All Areas Are Equal: Transfer Learning for Semantic Segmentation via Hierarchical Region Selection
- [2019CVPR] Beyond Gradient Descent for Regularized Segmentation Losses [[Code]](https://github.com/dmitrii-marin/adm-seg)
- [2019CVPR] Co-occurrent Features in Semantic Segmentation
- [2019CVPR] Context-aware Spatio-recurrent Curvilinear Structure Segmentation [line structure seg]
- [2019CVPR] Dual attention network for scene segmentation
- [2019CVPR] Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation.
- [2019AAAI] Learning Fully Dense Neural Networks for Image Semantic Segmentation
- [2019MICCAI] ET-Net: A Generic Edge-Attention Guidance Network for Medical Image Segmentation [[Code]](https://github.com/Yuju-arch/ETNet)
- [2019MICCAI] Attention Guided Network for Retinal Image Segmentation [[Code]](https://github.com/HzFu/AGNet)
- [2019MICCAIW] CU-Net: Cascaded U-Net with Loss Weighted Sampling for Brain Tumor Segmentation

- [2018CVPR] [EncNet] Context Encoding for Semantic Segmentation (oral) [[Code-Pytorch]](https://hangzhang.org/PyTorch-Encoding/experiments/segmentation.html) [[Slides]](https://hangzhang.org/slides/EncNet_slides.pdf)
- [2018CVPR] Learning a Discriminative Feature Network for Semantic Segmentation
- [2018CVPR] **DenseASPP** for Semantic Segmentation in Street Scenes [[Code]](https://github.com/DeepMotionAIResearch/DenseASPP)
- [2018CVPR] Dense Decoder Shortcut Connections for Single-Pass Semantic Segmentation

- [2018ECCV] **ESPNet**: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

- [2018ECCV] **ICNet** for Real-Time Semantic Segmentation on High-Resolution Images [[Proj]](https://hszhao.github.io/projects/icnet/) [[Code]](https://github.com/hszhao/ICNet)
- [2018ECCV] **PSANet**: Point-wise Spatial Attention Network for Scene Parsing
- [2018ECCV] **Bisenet**: Bilateral segmentation network for real-time semantic segmentation [[Code]](https://github.com/CoinCheung/BiSeNet)
- [2018ECCV] [**DeepLabv3+**] Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Code]](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [2018BMVC] Pyramid Attention Network for Semantic Segmentation
- [2018DLMIA] UNet++: A Nested U-Net Architecture for Medical Image Segmentation [[Code]](https://github.com/MrGiovanni/UNetPlusPlus)
- [2018MIDL] Attention U-Net: Learning Where to Look for the Pancreas
- [2017arXiv] [**DeepLabv3**] Rethinking Atrous Convolution for Semantic Image Segmentation
- [2017PAMI] [**DeepLabv2**] DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
- [2017PAMI] **SegNet**: A deep convolutional encoder-decoder architecture for image segmentation
- [2017CVPR] [**GCN**] Large Kernel Matters-Improve Semantic Segmentation by Global Convolutional Network [[Code]](https://github.com/SConsul/Global_Convolutional_Network) [[Note]](https://towardsdatascience.com/review-gcn-global-convolutional-network-large-kernel-matters-semantic-segmentation-c830073492d2)
- [2017CVPR] [**PSPNet**] Pyramid Scene Parsing Network
- [2017CVPR] **RefineNet**: Multi-path refinement networks for high-resolution semantic segmentation
- [2017CVPR] [**FCIS**] Fully convolutional instance-aware semantic segmentation
- [2017CVPR] [**FRRN**] Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes [[Code]](https://github.com/TobyPDE/FRRN)
- [2017CVPRW] The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation [[Code]](https://github.com/SimJeg/FC-DenseNet)

- [2017ICRA] **AdapNet**: Adaptive semantic segmentation in adverse environmental conditions [[Code]](https://github.com/DeepSceneSeg/AdapNet)
- [2016ICLR] Multi-Scale Context Aggregation by Dilated Convolutions
- [2016ICLR] ParseNet: Looking Wider to See Better
- [2016CVPR] Instance-aware semantic segmentation via multi-task network cascades
- [2016CVPR] Attention to Scale: Scale-Aware Semantic Image Segmentation

- [2016ECCV] What's the Point: Semantic Segmentation with Point Supervision
- [2016ECCV] Instance-sensitive fully convolutional networks

- [2016DLMIA] [UNet+ResNet] The Importance of Skip Connections in Biomedical Image Segmentation
- [2015ICLR] [**DeepLabv1**] Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
- [2015ICCV] Conditional random fields as recurrent neural networks
- [2015ICCV] [DeconvNet] Learning Deconvolution Network for Semantic Segmentation
- [2015MICCAI] **U-Net**: Convolutional networks for biomedical image segmentation [[Note]](https://mp.weixin.qq.com/s/dPe9HyQuKQr_C9jb9QqyXQ)
- [2015CVPR/2017PAMI] [**FCN**] Fully convolutional networks for semantic segmentation 

**PanopticSeg**

- > [awesome-panoptic-segmentation](https://github.com/Angzz/awesome-panoptic-segmentation)

- Real-Time Panoptic Segmentation from Dense Detections, arXiv2019.12

- Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation, [arXiv2019.12](https://arxiv.org/abs/1911.10194)

- PanDA: Panoptic Data Augmentation, arXiv2019.11

- Learning Instance Occlusion for Panoptic Segmentation, arXiv2019.11

- Panoptic Edge Detection, arXiv2019.6

- [2020ICRA] **DS-PASS**: Detail-Sensitive Panoramic Annular Semantic Segmentation through SwaftNet for Surrounding Sensing [[Code]](https://github.com/elnino9ykl/DS-PASS)

- [2020AAAI] **SOGNet**: Scene Overlap Graph Network for Panoptic Segmentation

- [2019CVPR] Panoptic Segmentation

- [2019CVPR] Attention-guided Unified Network for Panoptic Segmentation

- [2019CVPR] Panoptic Feature Pyramid Networks (oral) [[unofficial code](https://github.com/Angzz/panoptic-fpn-gluon)] [[detectron2]](https://github.com/facebookresearch/detectron2)

- [2019CVPR] **UPSNet**: A Unified Panoptic Segmentation Network [[Code]](https://github.com/uber-research/UPSNet)

- [2019CVPR] [**OANet**] An End-to-end Network for Panoptic Segmentation

- [2019CVPR] **DeeperLab**: Single-Shot Image Parser (oral) [[project]](http://deeperlab.mit.edu/) [[code]](https://github.com/tensorflow/models/tree/master/research/deeplab/evaluation)

- [2019CVPR] Interactive Full Image Segmentation by Considering All Regions Jointly

- [2019CVPR] Seamless Scene Segmentation [[code]](https://github.com/mapillary/seamseg)



## 3DVision

> [awesome image-based 3D reconstruction](https://github.com/openMVG/awesome_3DReconstruction_list)
>
> [awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis)
>
> [Blog] [基于单目视觉的三维重建算法综述](https://cloud.tencent.com/developer/article/1397509)
>
> [Bolg] [三维视觉、SLAM方向全球顶尖实验室汇总](https://mp.weixin.qq.com/s/09jdFSlhHwakMAGVxha3NA)
>
> Camera Calibration [[Note]](https://mp.weixin.qq.com/s/AU9uLn6cncgjD5I8r_mUvQ) [[Note2]](https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247486940&idx=1&sn=5a00a823dfaa7cafeda5a5d74394cb1f&chksm=97d7e84ba0a0615d48a2dad237449cfcea93430e7af2ab035775382ac667f2b1de0e91e8cc18&mpshare=1&scene=24&srcid=02135YhWNDRG2lgQFtnxoGkY#rd)
>
> [Hub] [Visual SLAM Related Research](https://github.com/wuxiaolang/Visual_SLAM_Related_Research)



### Review

- > BigSFM: Reconstructing the World from Internet Photos, summary of Noah Snavely works [[Proj&Code]](http://www.cs.cornell.edu/projects/bigsfm/) (Bundler, 1DSfM, sfm-dismbig, DISCO, LocalSymmetry, dataset ...)
  >

- A Survey on Deep Leaning Architectures for Image-based Depth Reconstruction, arXiv2019.6

- [2019PAMI] Image-based 3D Object Reconstruction: State-of-the-Art and Trends in the Deep Learning Era

- [2017Robot] Keyframe-based monocular SLAM: design, survey, and future directions, Robotics and Autonomous Systems

  

### CameraPose

- [2017CVPR] Geometric loss functions for camera pose regression with deep learning [[Proj-with PoseNet+Modelling]](http://mi.eng.cam.ac.uk/projects/relocalisation/#results)
- [2016ICRA] Modelling Uncertainty in Deep Learning for Camera Relocalization 
- [2015ICCV] PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization 



### SfM

- [2016ECCV] [**LineSfM**] Robust and Accurate Line- and/or Point-Based Pose Estimation without Manhattan Assumptions [[Code]](https://github.com/ySalaun/LineSfM)



### Depth/StereoMatching

- [2020PAMI] SurfaceNet+: An End-to-end 3D Neural Network for Very Sparse Multi-view Stereopsis [[Code]](https://github.com/mjiUST/SurfaceNet-plus)
- [2020ICLR] **Pseudo-LiDAR++**: Accurate Depth for 3D Object Detection in Autonomous Driving, arXiv2019.8 [[Code]](https://github.com/mileyan/Pseudo_Lidar_V2)
- [2019NIPS] [**SC-SfMLearner**] Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video [[Proj]](https://jwbian.net/sc-sfmlearner) [[Code]](https://github.com/JiawangBian/SC-SfMLearner-Release)
- [2019ICCV] How do neural networks see depth in single images? [[Note]](https://zhuanlan.zhihu.com/p/95758284)
- [2019ICCV] **DeepPruner**: Learning Efficient Stereo Matching via Differentiable PatchMatch [[Code]](https://github.com/uber-research/DeepPruner)
- [2019CVPR] **Pseudo-LiDAR** from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving [[Code]](https://github.com/mileyan/pseudo_lidar)
- [2019CVPR] **DeepLiDAR**: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image
- [2019CVPR] [**R-MVSNet**] Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference [[Code]](https://github.com/YoYo000/MVSNet)
- [2019ToG] 3D Ken Burns Effect from a Single Image [[Homepage]](https://sniklaus.com/papers/kenburns) [[Code]](https://github.com/sniklaus/3d-ken-burns)
- [2019IROS] **SuMa++**: Efficient LiDAR-based Semantic SLAM [[Code]](https://github.com/PRBonn/rangenet_lib)
- [2019ICCVW] Self-Supervised Learning of Depth and Motion Under Photometric Inconsistency
- [2019WACV] **SfMLearner++**: Learning Monocular Depth & Ego-Motion using Meaningful Geometric Constraints
- [2018CVPR] Automatic 3D Indoor Scene Modeling From Single Panorama
- [2018CVPR] **LEGO**: Learning Edge with Geometry all at Once by Watching Videos (spotlight) [[Code]](https://github.com/zhenheny/LEGO)
- [2018CVPR] [**vid2depth**] Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints [[Proj&Code]](https://sites.google.com/view/vid2depth)
- [2018CVPR] **GeoNet**: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose [[Code]](https://github.com/yzcjtr/GeoNet)
- [2018CVPR] **DeepMVS**: Learning Multi-View Stereopsis [[Proj]](https://phuang17.github.io/DeepMVS/index.html) [[Code]](https://github.com/phuang17/DeepMVS)
- [2018ECCV] **MVSNet**: Depth Inference for Unstructured Multi-view Stereo
- [2017ICCV] **SurfaceNet**: An End-to-end 3D Neural Network for Multiview Stereopsis [[Code]](https://github.com/mjiUST/SurfaceNet)
- [2017CVPR] [**SfMLearner**] Unsupervised Learning of Depth and Ego-Motion from Video, Oral [[Proj]](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/)  [[TF]](https://github.com/tinghuiz/SfMLearner) [[Pytorch]](https://github.com/ClementPinard/SfmLearner-Pytorch) [[ClassProj]](https://cmsc733.github.io/2019/proj/p4/)
- [2017CVPR] **SGM-Nets**: Semi-Global Matching With Neural Networks
- [2016JMLR] [**MC-CNN**] Stereo matching by training a convolutional neural network to compare image patches [[Code]](https://github.com/jzbontar/mc-cnn)



### Surface Reconstruction 

- [ICCV15/IJCV17] Global, Dense Multiscale Reconstruction for a Billion Points [[Proj]](https://lmb.informatik.uni-freiburg.de/people/ummenhof/multiscalefusion/) [[Code]](https://lmb.informatik.uni-freiburg.de/people/ummenhof/multiscalefusion/)
- [2014ECCV] Let there be color! Large-scale texturing of 3D reconstructions [[Code]](https://github.com/nmoehrle/mvs-texturing)



### 3D Layout

- [2017WACV] Pano2CAD: Room Layout From A Single Panorama Image

- [2014ECCV] **PanoContext**: A Whole-room 3D Context Model for Panoramic Scene Understanding, Oral [[Homepage&Code]](http://panocontext.cs.princeton.edu/) [[PanoBasic]](https://github.com/yindaz/PanoBasic)



### 3D SemanticSeg

- Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping, [arXiv2019.12](https://arxiv.org/abs/1910.02490) [[Code]](https://github.com/MIT-SPARK/Kimera-Semantics)

- Rotation Invariant Point Cloud Classification: Where Local Geometry Meets Global Topology, [arXiv2019.11](https://arxiv.org/abs/1911.00195)

- SalsaNet: Fast Road and Vehicle Segmentation in LiDAR Point Clouds for Autonomous Driving, [arXiv2019.9](https://arxiv.org/abs/1909.08291) [[Code]](https://gitlab.com/aksoyeren/salsanet)

- Going Deeper with Point Networks, [arXiv2019.7](https://arxiv.org/abs/1907.00960) [[Code]](https://github.com/erictuanle/GoingDeeperwPointNetworks)

- [2020GRSM] A Review of Point Cloud Semantic Segmentation

- [2019NIPS] [**PVCNN**] Point-Voxel CNN for Efficient 3D Deep Learning (Spotlight) [[Proj]](https://hanlab.mit.edu/projects/pvcnn/) [[Code]](https://github.com/mit-han-lab/pvcnn)

- [2019IROS] **RangeNet++**: Fast and Accurate LiDAR Semantic Segmentation [[Code]](https://github.com/PRBonn/lidar-bonnetal)

- [2019ICCV] SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences

- [2019ICCV] Hierarchical Point-Edge Interaction Network for Point Cloud Semantic Segmentation

- [2019ICCV] Cascaded Context Pyramid for Full-Resolution 3D Semantic Scene Completion (oral)

- [2019CVPR] **ClusterNet**: Deep Hierarchical Cluster Network With Rigorously Rotation-Invariant Representation for Point Cloud Analysis

- [2018NIPS] **PointCNN**: Convolution On X-Transformed Points [[Code]](https://github.com/yangyanli/PointCNN)

- [2018ECCV] Efficient Semantic Scene Completion Network with Spatial Group Convolution [[Code]](https://github.com/zjhthu/SGC-Release)

- [2017NIPS] **PointNet++**: Deep Hierarchical Feature Learning on Point Sets in a Metric Space [[Code]](https://github.com/charlesq34/pointnet2)

- [2017CVPR] **PointNet**: Deep Learning on Point Sets for 3D Classification and Segmentation [[Code]](https://github.com/charlesq34/pointnet)

  

## LowLevelVision

**Tutorial&Reviews**

- ICCV2019 Tutorial: Understanding Color and the In-Camera Image Processing Pipeline for Computer Vision, Michael S. Brown [[Homepage]](https://www.eecs.yorku.ca/~mbrown/ICCV2019_Brown.html) [[Slides]](docs/Michael_S_Brown_ICCV19_Tutorial)
- CVPR2016 Tutorial: Understanding the In-Camera Image Processing Pipeline for Computer Vision, Michael S. Brown [[Slides]](docs/Michael_S_Brown_Tutorial_CVPR2016)
- NIPS2011 Tutorial: Modeling the Digital Camera Pipeline: From RAW to sRGB and Back, Michael S Brown [[Slides]](docs/Michael_S_Brown_Modeling_Digital_Camera_NIPS2011)

**RAW**

- [2018IJCV] RAW Image Reconstruction Using a Self-contained sRGB–JPEG Image with Small Memory Overhead [Michael S. Brown]
- [2016CVPR] RAW Image Reconstruction using a Self-Contained sRGB-JPEG Image with only 64 KB Overhead
- [2014CVPR] Raw-to-raw: Mapping between image sensor color responses

**Super-Resolution**

- > [Blog] [深入浅出深度学习超分辨率](https://mp.weixin.qq.com/s/o-I6T8f4AcETJqlDNZs9ug

- A Deep Journey into Super-resolution: A survey, arXiv2019.9
- [2020PAMI] Deep Learning for Image Super-resolution: A Survey
- [2019IJAC] Deep Learning Based Single Image Super-resolution: A Survey
- 
- Densely Residual Laplacian Super-resolution, [arXiv2019.7](https://arxiv.org/abs/1906.12021) [[Code]](https://github.com/saeed-anwar/DRLN)
- Lightweight Image Super-Resolution with Adaptive Weighted Learning Network, [arXiv2019.4](https://arxiv.org/abs/1904.02358) [[Code]](https://github.com/ChaofWang/AWSRN)
- [2019SIGG] Handheld Multi-Frame Super-Resolution

- [2019CVPR] Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels
- [2019CVPR] Zoom To Learn, Learn To Zoom [[ProjPage]](https://ceciliavision.github.io/project-pages/project-zoom.html) [[Code]](https://github.com/ceciliavision/zoom-learn-zoom) 
- [2019CVPR] Towards Real Scene Super-Resolution with Raw Images [[Code]](https://github.com/git-davi/Towards-Real-Scene-Super-Resolution-with-Raw-Images)
- [2019CVPR] 3D Appearance Super-Resolution with Deep Learning [[Code]](https://github.com/ofsoundof/3D_Appearance_SR)
- [2019CVPR] Learning Parallax Attention for Stereo Image Super-Resolution [[Code]](https://github.com/LongguangWang/PASSRnet)
- [2019CVPR] Meta-SR: A Magnification-Arbitrary Network for Super-Resolution [[github]](https://github.com/XuecaiHu/Meta-SR-Pytorch)
- [2019CVPRW] Hierarchical Back Projection Network for Image Super-Resolution [[Code]](https://github.com/Holmes-Alan/HBPN)
- [2019ICCVW] Edge-Informed Single Image Super-Resolution [[Code]](https://github.com/knazeri/edge-informed-sisr)
- [2017CVPRW] Enhanced Deep Residual Networks for Single Image Super-Resolution [[Code]](https://github.com/thstkdgus35/EDSR-PyTorch)
- [2016PAMI] [**SRCNN**] Image Super-Resolution Using Deep Convolutional Networks
- [2016NIPS] Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections
- [2016CVPR] [**ESPCN**] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
- [2016CVPR] [**VDSR**] Accurate Image Super-Resolution Using Very Deep Convolutional Networks
- [2016ECCV] [**FSRCNN**] Accelerating the Super-Resolution Convolutional Neural Network
- [2014ECCV] [**SRCNN**] Learning a Deep Convolutional Network for Image Super-Resolution

**Enhancement**

- Diving Deeper into Underwater Image Enhancement: A Survey, arXiv2019.7
- 
- [2018CVPR] Deep Photo Enhancer: Unpaired Learning for Image Enhancement from Photographs with GANs,  [[Homepage]](http://www.cmlab.csie.ntu.edu.tw/project/Deep-Photo-Enhancer/) [[Code]](https://github.com/nothinglo/Deep-Photo-Enhancer) 
- [2018CVPR] Classification-Driven Dynamic Image Enhancement
- [2017CVPR] Forget Luminance Conversion and Do Something Better
- [2016CVPR] Two Illuminant Estimation and User Correction Preference
- 
- Low-light Enhancement Repo [[github]](https://github.com/rockeyben/Low-Light)
- 基于深度学习的低光照图像增强方法总结(2017-2019) [[Note]](https://zhuanlan.zhihu.com/p/78297097)
- Learning to see, Antonio Torralba, 2016 [[Slides]](https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649286338&idx=1&sn=e5ed56e3fba3ef44af3d593dec55d764&scene=1&srcid=0827T9GPh4bd2NQ1BEgCocD5#rd)
- Attention-guided Low-light Image Enhancement, arXiv2019.8 
- Low-light Image Enhancement Algorithm Based on Retinex and Generative Adversarial Network, arXiv2019.6
- LED2Net: Deep Illumination-aware Dehazing with Low-light and Detail Enhancement, arXiv2019.6
- EnlightenGAN: Deep Light Enhancement without Paired Supervision, arXiv2019.6 [[Code]](https://github.com/TAMU-VITA/EnlightenGAN)
- Kindling the Darkness: A Practical Low-light Image Enhancer, arXiv2019.5
- MSR-net: Low-light Image Enhancement Using Deep Convolutional Network, arXiv2017.11
- [2019TOG] Handheld Mobile Photography in Very Low Light
- [2019ICCV] Learning to See Moving Objects in the Dark
- [2019CVPR] Underexposed Photo Enhancement using Deep Illumination Estimation [[Code]](https://github.com/wangruixing/DeepUPE)
- [2019CVPR] All-Weather Deep Outdoor Lighting Estimation
- [2019MMM] Progressive Retinex: Mutually Reinforced Illumination-Noise Perception Network for Low Light Image Enhancement
- [2018TIP] Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images
- [2018TMM] Naturalness preserved nonuniform illumination estimation for image enhancement based on retinex
- [2018PRL] LightenNet: A Convolutional Neural Network for weakly illuminated image enhancement
- [2018CVPR] Learning to See in the Dark
- [2018BMVC] MBLLEN: Low-light Image/Video Enhancement Using CNNs
- [2018BMVC] Deep Retinex Decomposition for Low-Light Enhancement
- [2018BMVC] Deep Retinex Decomposition for Low-Light Enhancement (Oral) [[Proj]](https://daooshee.github.io/BMVC2018website/) [[Code]](https://github.com/weichen582/RetinexNet)
- [2017TIP] LIME: Low-light image enhancement via illumination map estimation
- [2017PR] LLNet: A deep autoencoder approach to natural low-light image enhancement [[Code]](https://github.com/pythonuser200/LLNet) [[Code2]](https://github.com/kglore/llnet_color)
- [2017CVPR] Deep Outdoor Illumination Estimation
- [2016TIP] LIME: Low-light Image Enhancement via Illumination Map Estimation
- [2016ECCV] Deep Specialized Network for Illuminant Estimation

**Reflection Removal**

- [2019CVPR] Single Image Reflection Removal Beyond Linearity
- [2019CVPR] Reflection Removal Using A Dual-Pixel Sensor 
- [2013ICCV] Exploiting Reflection Change for Automatic Reflection Removal

**Denoising**

- Deep Learning on Image Denoising: An overview, [arXiv2020.1](https://arxiv.org/abs/1912.13171) [[Proj]](https://github.com/hellloxiaotian/Deep-Learning-on-Image-Denoising-An-overview)
- [2020NN] Attention-guided CNN for image denoising [[Code]](https://github.com/hellloxiaotian/ADNet)
- [2019CVPR] Toward Convolutional Blind Denoising of Real Photographs

**Deblurring**

- [SelfDeblur] Neural Blind Deconvolution Using Deep Priors, arXiv2019. 8 [[Code]](https://github.com/csdwren/SelfDeblur)
- [2019ICCV] DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better [[Code]](https://github.com/TAMU-VITA/DeblurGANv2)
- [2018CVPR] DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks [[Code]](https://github.com/KupynOrest/DeblurGAN)
- [2018CVPR] Dynamic Scene Deblurring Using Spatially Variant Recurrent Neural Networks 

**Deraining**

- [Single Image Deraining](https://paperswithcode.com/task/single-image-deraining/codeless) [Rain Removal](https://paperswithcode.com/task/rain-removal/codeless)
- [2019CVPR] Single Image Deraining: A Comprehensive Benchmark Analysis
- [2018ECCV] Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining [[Code]](https://github.com/XiaLiPKU/RESCAN)

**Completion**

- Image inpainting: A review, [arXiv2019.9](https://arxiv.org/abs/1909.06399)
- Consistent Generative Query Networks, [arXiv2019.4](https://arxiv.org/abs/1807.02033) [[Proj]](https://docs.google.com/document/d/1fSHMTQrH01KWuDVTKfr2aaUK2t4Am9Wv8cglmXRdKhM/edit)
- [2019Scirobotics] Emergence of exploratory look-around behaviors through active observation completion [[Proj]](http://vision.cs.utexas.edu/projects/visual-exploration/)
- [2019ICCV] An Internal Learning Approach to Video Inpainting [[Homepage]](https://cs.stanford.edu/~haotianz/publications/video_inpainting/#) [[Code]](https://github.com/Haotianz94/IL_video_inpainting) [[Note]](https://mp.weixin.qq.com/s?__biz=MzIwMTE1NjQxMQ==&mid=2247488623&idx=2&sn=7c80f4d6da84e6187659d8dbfb3c6562&chksm=96f3663ba184ef2d3aa8b36f594ba05546bffeb0b843cbaa2f94f8f5c3ac0342768bf704ce5f&mpshare=1&scene=23&srcid=0925gCwwwBVGAqYZ5fM6pXhR&sharer_sharetime=1574684675459&sharer_shareid=d3d8827bce826478944c5e3a9f67ed4b%23rd)
- [2019ICCV] StructureFlow: Image Inpainting via Structure-aware Appearance Flow [[Code]](https://github.com/RenYurui/StructureFlow)
- [2018Science] [GQN] Neural scene representation and rendering, DeepMind [[Code]](https://github.com/wohlert/generative-query-network-pytorch) [[Note]](https://zhuanlan.zhihu.com/p/38132269)
- [2018CVPR] Learning to Look Around: Intelligently Exploring Unseen Environments for Unknown Tasks [[Code]](https://github.com/dineshj1/lookaround) 
- [2018CVPR] Deep Image Prior [[github]](https://dmitryulyanov.github.io/deep_image_prior) [[Note]](https://zhuanlan.zhihu.com/p/31595192)
- [2018Proj] Painting outside the box: image outpainting with GANs, Mark Sabini, Stanford CS230 Project, [arXiv2018.8](https://arxiv.org/abs/1808.08483) [[Code]](https://github.com/bendangnuksung/Image-OutPainting) [[PDF]](https://cs230.stanford.edu/projects_spring_2018/posters/8265861.pdf) [[Model]](https://drive.google.com/file/d/1548iAtsNf3wLSc1i5zYy-HX8_TW95wi_/view) [[Note]](http://www.sohu.com/a/244142600_473283)

**Image/Video Transfer**

- > Style Transfer Scholar: [Dongdong Chen](http://www.dongdongchen.bid/) [Dmitry Ulyanov](https://dmitryulyanov.github.io/about)

- [2018TOG] Progressive Color Transfer with Dense Semantic Correspondences  ⭐️⭐️⭐️⭐️
- [2017CVPR] Improved Texture Networks: Maximizing Quality and Diversity in Feed-forward Stylization and Texture Synthesis
- [2016ICML] Texture Networks: Feed-forward Synthesis of Textures and Stylized Images [IN] [[Code]](https://github.com/DmitryUlyanov/texture_nets) [[Slides]](/docs/TextureNetworks-talk.pdf)
- [2016CVPR] Image Style Transfer Using Convolutional Neural Networks, Gatys  [[Code]](https://github.com/Kautenja/a-neural-algorithm-of-artistic-style)
- [2016ECCV] Perceptual Losses for Real-Time Style Transfer and Super-Resolution
- [2015] A neural algorithm of artistic style, Gatys, [arXiv2015.9](https://arxiv.org/abs/1508.06576) [[Code]](https://github.com/Kautenja/a-neural-algorithm-of-artistic-style)

**Blending/Fusion**

- Deep Image Blending, [arXiv201910](https://arxiv.org/abs/1910.11495) [[Code]](https://github.com/owenzlz/Deep_Image_Blending)
- [2019MMM] GP-GAN: Towards Realistic High-Resolution Image Blending [[Code]](https://github.com/wuhuikai/GP-GAN) [[Homepage]](http://wuhuikai.me/GP-GAN-Project/)
- [2018ECCV] Learning to Blend Photos [[Homepage]](https://github.com/hfslyc/LearnToBlend)
- [2018SIGGA] Deep Blending for Free-Viewpoint Image-Based Rendering [[Homepage]](https://www-sop.inria.fr/reves/Basilic/2018/HPPFDB18/)



## Pedestrain/Crowd

**PedestrainDetection**

- > [Pedestrain Detection collection](https://github.com/xingkongliang/Pedestrian-Detection)

- Deep Learning for Person Re-identification: A Survey and Outlook, [arXiv2020.1](https://arxiv.org/abs/2001.04193) [[Code]](https://github.com/mangye16/ReID-Survey) 
- Pedestrain Attribute Recognition: A Survey, [arXiv2019.1](https://arxiv.org/abs/1901.07474) [[Proj]](https://sites.google.com/view/ahu-pedestrianattributes/)
- CrowdHuman: A Benchmark for Detecting Human in a Crowd, [arXiv201804](https://arxiv.org/abs/1805.00123) [[Proj]](http://www.crowdhuman.org/) [[Note]](https://github.com/Mycenae/PaperWeekly/blob/master/CrowdHuman.md)
- 
- PedHunter: Occlusion Robust Pedestrian Detector in Crowded Scenes, [arXiv2019.9](https://arxiv.org/abs/1909.06826)
- [2020TMM/2019CVPRW] Bag of Tricks and A Strong Baseline for Deep Person Re-identification [[Code]](https://github.com/michuanhaohao/reid-strong-baseline)
- [2019ICCV] Mask-Guided Attention Network for Occluded Pedestrian Detection [[Code]](https://github.com/Leotju/MGAN)
- [2019CVPR] VRSTC: Occlusion-Free Video Person Re-Identification [*occlusion*]
- [2018CVPR] Repulsion Loss: Detecting Pedestrians in a Crowd, CVPR2018 [*occlusion*]
- [2016ECCV] Stacked Hourglass Networks for Human Pose Estimation

**CrowdCounting**

- > [awesome crowd counting](https://github.com/gjy3035/Awesome-Crowd-Counting)

- Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection, [arXiv2019.6](https://arxiv.org/abs/1906.07538) [[Code]](https://github.com/val-iisc/lsc-cnn)

- W-Net: Reinforced U-Net for Density Map Estimation, [arXiv2019.3](https://arxiv.org/abs/1903.11249) [[Unofficial Code]](https://github.com/ZhengPeng7/W-Net-Keras)

- [2019TIP] HA-CCN: Hierarchical Attention-based Crowd Counting Network

- [2019ICCV] Bayesian Loss for Crowd Count Estimation with Point Supervision [[Code]](https://github.com/ZhihengCV/Bayesian-Crowd-Counting)

- [2019ICCV] Crowd Counting with Deep Structured Scale Integration Network (oral) [[github]](https://github.com/Legion56/Counting-ICCV-DSSINet)

- [2019ICCV] Learning Spatial Awareness to Improve Crowd Counting (oral)

- [2019ICCV] Perspective-Guided Convolution Networks for Crowd Counting [[Code]](https://github.com/Zhaoyi-Yan/PGCNet_pytorch) [[Dataset]](https://ai.baidu.com/broad/subordinate?dataset=crowd_surv)

- [2019ICCV] Learn to Scale: Generating Multipolar Normalized Density Maps for Crowd Counting

- [2019ICCV] Pushing the Frontiers of Unconstrained Crowd Counting: New Dataset and Benchmark Method

- [2019ICCV] Counting with Focus for Free [[Code]](https://github.com/shizenglin/Counting-with-Focus-for-Free)

- [2019ICCVW] Crowd Counting on Images with Scale Variation and Isolated Clusters

- [2019CVPR] Learning from Synthetic Data for Crowd Counting in the Wild [[Homepage]](https://gjy3035.github.io/GCC-CL/) [[Dataset]](https://github.com/gjy3035/GCC-CL)

- [2019MMM] Improving the Learning of Multi-column Convolutional Neural Network for Crowd Counting

- [2019ICME] Locality-constrained Spatial Transformer Network for Video Crowd Counting

- [2019SciAdvance] Number detectors spontaneously emerge ina deep neural network designed for visual object recognition [[Note]](https://zhuanlan.zhihu.com/p/65630935)

- [2019TII] Automated Steel Bar Counting and Center Localization with Convolutional Neural Networks [[Code]](https://github.com/BenzhangQiu/Steel-bar-Detection)
- [2018MICCAIW] Microscopy Cell Counting with Fully Convolutional Regression Networks [[Code]](https://github.com/WeidiXie/cell_counting_v2)
- [2010NIPS] Learning to count objects in images [[Code]](https://github.com/dwaithe/U-net-for-density-estimation)



## GenerativeNet

> GAN学习路线图：论文、应用、课程、书籍大总结 [[Page]](https://mp.weixin.qq.com/s?__biz=MzA4NzE1NzYyMw==&mid=2247499827&idx=2&sn=40255546e84e3b56582acc55e3bb84c5&scene=21#wechat_redirect)
>
> [深度学习中最常见GAN模型概览](https://mp.weixin.qq.com/s/Dlnve-n8oKfU3lwsdO2j2w): GAN,DCGAN,CGAN,infoGAN,ACGAN,CycleGAN,StackGAN ...
>
> [Blog: One Day One GAN](https://github.com/OUCMachineLearning/OUCML/tree/master/One_Day_One_GAN)



**Training Tricks**

- How to Train a GAN? Tips and tricks to make GANs work [[Page]](https://github.com/soumith/ganhacks)

  Start from NIPS2016, 17 GAN tricks, by Soumith Chintala, Emily Denton, Martin Arjovsky, Michael Mathieu.  How to Train a GAN, NeurIPS2016

- Top highlight Advances in Generative Adversarial Networks (GANs): A summary of the latest advances in Generative Adversarial Networks [[Page]](https://medium.com/beyondminds/advances-in-generative-adversarial-networks-7bad57028032) [[Note]](https://mp.weixin.qq.com/s/xUflzrv2Zi_dG3WXmdUKEw)

- Keep Calm and train a GAN. Pitfalls and Tips on training Generative Adversarial Networks [[Page]](https://mp.weixin.qq.com/s/3XihII4sRfE-TmGhP4h1pg)

**Papers**

- > [Blogg] [A Beginner's Guide to Generative Adversarial Networks (GANs)](https://pathmind.com/wiki/generative-adversarial-network-gan), 2019

- Generative Adversarial Networks: A Survey and Taxonomy, [arXiv2020.2](https://arxiv.org/abs/1906.01529) [[GANReview]](https://github.com/sheqi/GAN_Review)

- [2019ACMCS] How Generative Adversarial Networks and Their Variants Work: An Overview

- -

- StarGAN v2: Diverse Image Synthesis for Multiple Domains. [arXiv201912](https://arxiv.org/abs/1912.01865) [[Code]](https://github.com/clovaai/stargan-v2)

- This dataset does not exist: training models from generated images, [arXiv2019.11](https://arxiv.org/abs/1911.02888)

- Landmark Assisted CycleGAN for Cartoon Face Generation. [arXiv201907](https://arxiv.org/abs/1907.01424)

- Maximum Entropy Generators for Energy-Based Models, [arXiv2019.5](https://arxiv.org/abs/1901.08508) [[Code]](https://github.com/ritheshkumar95/energy_based_generative_models)

- [2019NIPS] Few-shot Video-to-Video Synthesis [[Code]](https://github.com/NVlabs/few-shot-vid2vid)

- [2019NIPS] [**vid2vid**] Video-to-Video Synthesis [[Code]](https://github.com/NVIDIA/vid2vid)

- [2019CVPR] Semantic Image Synthesis with Spatially-Adaptive Normalization [[Proj]](https://nvlabs.github.io/SPADE/) [[Code]](https://github.com/NVLabs/SPADE)

- [2019CVPR] [**seg2vid**] Video Generation from Single Semantic Label Map [[Code]](https://github.com/junting/seg2vid)

- [2019BMVC] The Art of Food: Meal Image Synthesis from Ingredients

- [2018ICLR] Spectral Normalization for Generative Adversarial Networks [[Code]](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan) [[Supp1]](https://christiancosgrove.com/blog/2018/01/04/spectral-normalization-explained.html) [[Supp2]](http://kaiz.xyz/blog/posts/spectral-norm/)

- [2018CVPR] [**pix2pixHD**] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs

- [2018CVPR] StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (oral) [[Code]](https://github.com/yunjey/stargan)

- [2018ECCV] [FE-GAN] Fashion Editing with Multi-scale Attention Normalization [[Notes]](https://mp.weixin.qq.com/s/APaI7eDhswIXr35_E5YcGw)

- [2018ECCV] Image Inpainting for Irregular Holes Using Partial Convolutions [[Code]](https://github.com/NVIDIA/partialconv) [[Code2]](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv) [[used for DeepNude]](https://github.com/yuanxiaosc/DeepNude-an-Image-to-Image-technology) 

- [2017ICCV] [**CycleGAN**] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[Proj]](https://junyanz.github.io/CycleGAN/)

- [2017CVPR] [**Pix2Pix**] Image-to-Image Translation with Conditional Adversarial Networks [[Demo]](https://affinelayer.com/pixsrv/)

- [2016ICLR] [**DCGAN**] Unsupervised representation learning with deep convolutional generative adversarial networks

- [2016ICML] A Theory of Generative ConvNet [S-C Zhu] [[Proj/Code]](http://www.stat.ucla.edu/~ywu/GenerativeConvNet/main.html)

- [2014NIPS] Generative Adversarial Nets



## Video Understanding

- [YOWO] You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization [arXiv201911](https://arxiv.org/abs/1911.06644) [[Code]](https://github.com/wei-tim/YOWO)

- [2019CVPR] Learning Video Representations from Correspondence Proposals

  > 现有的视频深度学习架构通常依赖于三维卷积、自相关、非局部模块等运算，这些运算难以捕捉视频中帧间的长程运动/相关性，该文提出的CPNet学习视频中图片之间的长程对应关系，来解决现有方法在处理视频长程运动中的局限性.

**Video Object Detection**

- Object Detection in Video with Spatial-temporal Context Aggregation, [arXiv2019.7](https://arxiv.org/abs/1907.04988)
- Looking Fast and Slow: Memory-Guided Mobile Video Object Detection, [arXiv2019.3](https://arxiv.org/abs/1903.10172) [[TF]](https://github.com/tensorflow/models/tree/master/research/lstm_object_detection) [[PyTorch]](https://github.com/vikrant7/pytorch-looking-fast-and-slow)
- [2019ICCV] [**MGAN**] Motion Guided Attention for Video Salient Object Detection
- [2019CVPR] Shifting More Attention to Video Salient Objection Detection [[Code]](https://github.com/DengPingFan/DAVSOD)
- [2019CVPR] Activity Driven Weakly Supervised Object Detection [[Code]](https://github.com/zhenheny/Activity-Driven-Weakly-Supervised-Object-Detection)
- [2019SysML] AdaScale: Towards Real-time Video Object Detection Using Adaptive Scaling
- [2019KDDW] Understanding Video Content: Efficient Hero Detection and Recognition for the Game "Honor of Kings" [Notes]](https://flashgene.com/archives/28803.html)
- [2018CVPR] Mobile Video Object Detection With Temporally-Aware Feature Maps

**Video Object segmentation**

- [2016CVPR] A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation

- 

- [2019ICCV] RANet: Ranking Attention Network for Fast Video Object Segmentation

- [2019CVPR] See More, Know More: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks [[Code]](https://github.com/carrierlxk/COSNet)

- [2019CVPR] Improving Semantic Segmentation via Video Propagation and Label Relaxation [[Code]](https://github.com/NVIDIA/semantic-segmentation)

  

## DeepCNN

- [Deep Learning Papers Reading Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)

### Optimization

> [Summary of SGD，AdaGrad，Adadelta，Adam，Adamax，Nadam](https://zhuanlan.zhihu.com/p/22252270)
>
> [Why Momentum Really Works, 2017](https://distill.pub/2017/momentum/)

- Optimization for deep learning: theory and algorithms. [arXiv201912](https://arxiv.org/abs/1912.08957) [[OptimizationCourse]]([Optimization Theory for Deep Learning](https://wiki.illinois.edu/wiki/display/IE598ODLSP19/IE598-ODL++Optimization+Theory+for+Deep+Learning))

- Why Adam Beats SGD for Attention Models. [arXiv201912](https://arxiv.org/abs/1912.03194)

- Momentum Contrast for Unsupervised Visual Representation Learning, Kaiming He [arXiv2019.11](https://arxiv.org/abs/1911.05722)

- Dynamic Mini-batch SGD for Elastic Distributed Training: Learning in the Limbo of Resources, Amazon, [arXiv2019.5](https://arxiv.org/abs/1904.12043)

- Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, [arXiv2018.4](https://arxiv.org/abs/1706.02677) [[Notes]](https://www.zhihu.com/question/60874090)

- [2019NIPS] Uniform convergence may be unable to explain generalization in deep learning

- [2019NIPS] Understanding the Role of Momentum in Stochastic Gradient Methods

- [2019NIPS] Lookahead optimizer: k steps forward, 1 step back [[Code]](https://github.com/michaelrzhang/lookahead) [[Pytorch]](https://github.com/alphadl/lookahead.pytorch) [[TF]](https://github.com/Janus-Shiau/lookahead_tensorflow)

- [2019ICLR] [AdaBound] Adaptive gradient methods with dynamic bound of learning rate [[Pytorch]](https://github.com/Luolc/AdaBound) [[TF-example]](https://github.com/taki0112/AdaBound-Tensorflow) 

  > AdaBound combines SGD and Adam to make it fast as Adam at training start and convergence like SGD later. Usage: require Python 3.6+, and pip install: pip install adabound, and then: optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1). Version of TensorFlow is coming.

- [2019CVPRW] The Indirect Convolution Algorithm

- [2019ISCAW] Accelerated CNN Training Through Gradient Approximation



### Training

> Fast training for neural networks, You Yang, Jiangmen Talk [[Video]](https://www.youtube.com/watch?v=8B7ywS1nlJY)
>
> [Training Tricks in Object Detection](http://spytensor.com/index.php/archives/53/)

- Student Specialization in Deep ReLU Networks With Finite Width and Input Dimension, [arXiv2019.11](https://arxiv.org/abs/1909.13458)
- Accelerating CNN Training by Sparsifying Activation Gradients, [arXiv2019.8](https://arxiv.org/abs/1908.00173)
- Luck Matters: Understanding Training Dynamics of Deep ReLU Networks, [arXiv2019.6](https://arxiv.org/abs/1905.13405) [[Code]](https://github.com/facebookresearch/luckmatters)
- Bag of Freebies for Training Object Detection Neural Networks, Amazon, [arXiv2019.4](https://arxiv.org/abs/1902.04103) [[Code]](https://github.com/dmlc/gluon-cv\)
- Deep Double Descent: Where Bigger Models and More Data Hurt, ICLR2020Review
- [2019ICCV] Rethinking ImageNet Pre-training, FAIR [[Notes]](https://zhuanlan.zhihu.com/p/86886887)
- [2019CVPR] Bag of Tricks for Image Classification with Convolutional Neural Networks, Amazon [[Code]](https://gluon-cv.mxnet.io/) [[Note]](https://www.jianshu.com/p/0e0bc5dc300a)
- [2019CVPR] Accelerating Convolutional Neural Networks via Activation Map Compression
- [2019CVPR] RePr: Improved Training of Convolutional Filters [[Note]](https://zhuanlan.zhihu.com/p/58095683?utm_source=wechat_timeline&utm_medium=social&utm_oi=697209491862593536&from=timeline)
- [2019BMVC] Dynamic Neural Network Channel Execution for Efficient Training
- [2018ICPP] Imagenet training in minutes

**Activation**

- > [Blog] [深度学习中的激活函数](https://cloud.tencent.com/developer/article/1548121)
  >
  > Dead Relu [[Notes]](https://zhuanlan.zhihu.com/p/67054518)

- [2019CVPR] Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem (oral) [[Code]](https://github.com/max-andr/relu_networks_overconfident)

- [2018] [**GELU**] Gaussian Error Linear Units (GELUs). [arXiv201811](https://arxiv.org/abs/1606.08415) [[Note]](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650777736&idx=1&sn=840b4ae247e4fc47790d65a6f2af5664&chksm=871a6cf6b06de5e0f97bf4b3d4bcc39352cb50eb09e12356c3efc41c060ec4566791c342b799&mpshare=1&scene=23&srcid&sharer_sharetime=1577681186728&sharer_shareid=d3d8827bce826478944c5e3a9f67ed4b%23rd)

  ```
  # GELU in GPT-2: def gelu(x): return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
  ```

- [2016ICML] [CReLU] Understanding and improving convolutional neural networks via concatenated rectified linear units

- [2015ICCV] [PReLU-Net/msra Initilization] Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

**Normalization**

- > Normalization Scholar: [Ping Luo](http://luoping.me/)
  >
  > [Blog] Introduction to Normalization [[Page]](https://zhuanlan.zhihu.com/p/43200897) [[Note]](https://kexue.fm/archives/6992) 
  >
  > [Blog] Introduction to BN/LN/IN/GN [[Page]](https://zhuanlan.zhihu.com/p/72589565) [[Page2]](https://zhuanlan.zhihu.com/p/69659844)
  >
  > [Talk] Devils in BatchNorm, Jiangmen Talk, 2019 [[Page]](https://www.bilibili.com/video/av60805995?from=search&seid=17829763828493062183)
  >
  > [Blog] An Overview of Normalization Methods in Deep Learning, 2018.11 [[Page]](https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)

- Attentive Normalization. [Tianfu Wu] [arXiv2019.11](https://arxiv.org/abs/1908.01259) [[Code]](https://github.com/Cyril9227/Keras_AttentiveNormalization)

- Network Deconvolution. [a alternative to Batch Normalization]. arXiv2019.9 [[Proj]](https://sites.google.com/view/cxy)

- Weight Standardization. [arXiv2019.3](https://arxiv.org/abs/1903.10520) [[Code]](https://github.com/joe-siyuan-qiao/WeightStandardization)

- [**IN**] Instance Normalization: The Missing Ingredient for Fast Stylization. [arXiv2017.11](https://arxiv.org/abs/1607.08022) [[Code]](https://github.com/DmitryUlyanov/texture_nets)

- [**LN**] Layer Normalization. [Hinton] [arXiv2016.7](https://arxiv.org/abs/1607.06450) [[Note]](https://leimao.github.io/blog/Layer-Normalization/)

- [2019NIPS] Understanding and Improving Layer Normalization

- [2019NIPS] Positional Normalization [[Code]](https://github.com/Boyiliee/PONO) [[Supp]](https://papers.nips.cc/paper/8440-positional-normalization)

- [2018NIPS] How Does Batch Normalization Help Optimization? [[arXiv19v]](https://arxiv.org/abs/1805.11604) [[Ref]](https://mp.weixin.qq.com/s/laj9ZFq29db3iykU9Y56Tg)

- [2018NIPS] [**BIN**] Batch-Instance Normalization for Adaptively Style-Invariant Neural Networks [[Code]](https://github.com/hyeonseobnam/Batch-Instance-Normalization)

- [2018ECCV] [**GN**] Group normalization

- [2017NIPS] Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models

- [2016NIPS] [**WN**] Weight normalization: A simple reparameterization to accelerate training of deep neural networks

- [2015ICML] [**BN**] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Dropout**

- > Dropout [[Note1]](https://mp.weixin.qq.com/s/s_hNJwrK7uAeHVLWKssyxg) [[Note2]](https://zhuanlan.zhihu.com/p/76636329)

- [2014JMLR] Dropout: a simple way to prevent neural networks from overfitting

- [2012NIPS] ImageNet Classification with Deep Convolutional Neural Networks

**Augmentation**

- > [Blog] [Research Guide: Data Augmentation for Deep Learning](https://mc.ai/research-guide-data-augmentation-for-deep-learning/). 201910
  >
  > [Blog] [Data Augmentation: How to use Deep Learning when you have Limited Data](https://www.kdnuggets.com/2018/05/data-augmentation-deep-learning-limited-data.html). 201805 [[Page]](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/) 

- [2019JBD] A survey on Image Data Augmentation for Deep Learning. [[PDF]](https://link.springer.com/article/10.1186/s40537-019-0197-0) [[Notes]](https://zhuanlan.zhihu.com/p/76044027)

- Data Augmentation Revisited: Rethinking the Distribution Gap between Clean and Augmented Data. [arXiv2019.11](https://arxiv.org/abs/1909.09148)

- 

- GridMask Data Augmentation. [arXiv202001](https://arxiv.org/abs/2001.04086) [[Code]](https://github.com/akuxcw/GridMask) [[Note]](https://mp.weixin.qq.com/s/TxZ8usl1gaflniXGB0xaew)

- Let’s Get Dirty: GAN Based Data Augmentation for Soiling and Adverse Weather Classification in Autonomous Driving. [arXiv2019.12](https://arxiv.org/abs/1912.02249)

- PanDA: Panoptic Data Augmentation, arXiv2019.11

- Faster AutoAugment: Learning augmentation strategies using backpropagation. [arXiv201911](https://arxiv.org/abs/1911.06987)

- Automatic Data Augmentation by Learning the Deterministic Policy. [arXiv201910](https://arxiv.org/abs/1910.08343)

- Greedy AutoAugment, arXiv2019.8

- Safe Augmentation: Learning Task-Specific Transformations from Data, arXiv2019.7 [[Code]](https://github.com/Irynei/SafeAugmentation)

- Learning Data Augmentation Strategies for Object Detection. [arXiv201906](https://arxiv.org/abs/1906.11172) [[Code]](https://github.com/tensorflow/tpu/tree/master/models/official/detection)

- [2020ICLR] **AugMix**: A Simple Data Processing Method to Improve Robustness and Uncertainty [[Code]](https://github.com/google-research/augmix)

- [2019NIPS] Implicit Semantic Data Augmentation for Deep Networks

- [2019NIPS] Fast AutoAugment

- [2019ICML] Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules [[Code]](https://github.com/arcelien/pba) [[Examples]](https://bair.berkeley.edu/blog/2019/06/07/data_aug/)

- [2019ICCV] **CutMix**: Regularization Strategy to Train Strong Classifiers with Localizable Features [[Code]](https://github.com/clovaai/CutMix-PyTorch)

- [2019ICCVW] Occlusions for Effective Data Augmentation in Image Classification

- [2019ICCVW] **Style Augmentation**: Data Augmentation via Style Randomization

- [2019CVPR] **AutoAugment**: Learning Augmentation Policies from Data [[Code]](https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/autoaugment_utils.py#L15)

- [2018ICLR] **Mixup**: Beyond empirical risk minimization

- [2018ACML] **RICAP**: Random Image Cropping and Patching Data Augmentation for Deep CNNs [[Code]](https://github.com/shunk031/chainer-RICAP)

- [2018ICANN] Further advantages of data augmentation on convolutional neural networks (best paper)



### Model

> [Blog] [从Softmax到AMSoftmax](https://zhuanlan.zhihu.com/p/97475133)
>
> [Blog] [Convolutional Neural Networks Structure](https://zhuanlan.zhihu.com/p/28749411)
>
> [Blog] [A Survey of the Recent Architectures of Deep Convolutional Neural Networks](https://cloud.tencent.com/developer/article/1419844), 2019
>
> [Blog] [CNN下/上采样详析](https://zhuanlan.zhihu.com/p/94477174)

- A closer look at network resolution for efficient network design. [arXiv201909](https://arxiv.org/abs/1909.12978) [[Code]](https://drive.google.com/file/d/1HbASxAn7L0Elp09bdWqAmyQoSJ-smxI_/view)
- [2019NIPS] Is Deeper Better only when Shallow is Good? [[Code]](https://github.com/emalach/IsDeeperBetter)
- [2015Nature] Deep Learning Review
- [2014BMVC] Return of the Devil in the Details: Delving Deep into Convolutional Nets

**Module**

- **Pooling:**

  ViP: Virtual Pooling for Accelerating CNN-based Image Classification and Object Detection, arXiv201906

  Learning Spatial Pyramid Attentive Pooling in Image Synthesis and Image-to-Image Translation, arXiv201901

  [2020AAAI] Revisiting Bilinear Pooling: A coding Perspective [[Note]](https://zhuanlan.zhihu.com/p/62532887)

  [2019ICCV] LIP: Local Importance-based Pooling [[Code]](https://github.com/sebgao/LIP) [[Notes]](https://zhuanlan.zhihu.com/p/85841067)

  [2018ECCV] Grassmann Pooling as Compact Homogeneous Bilinear Pooling for Fine-Grained Visual Classification

  [2017CVPR] Low-rank bilinear pooling for fine-grained classification

  [2016EMNLP] Multimodal compact bilinear pooling for visual question answering and visual grounding

  [2016CVPR] Compact bilinear pooling

  [2015ICCV] [bilinear pooling] Bilinear CNN Models for Fine-grained Visual Recognition

  [2012ECCV] Semantic segmentation with second-order pooling

- Dynamic Convolutions: Exploiting Spatial Sparsity for Faster Inference [arXiv201912](https://arxiv.org/abs/1912.03203)
- Rethinking Softmax with Cross-Entropy: Neural Network Classifier as Mutual Information Estimator, arXiv201911
- Rethinking the Number of Channels for the Convolutional Neural Network, arXiv201909 
- AutoGrow: Automatic Layer Growing in Deep Convolutional Networks, arXiv201909 [[Code]](https://github.com/wenwei202/autogrow)
- Mapped Convolutions. [For 2D/3D/Spherical]. [arXiv201906](https://arxiv.org/abs/1906.11096) [[Code]](https://github.com/meder411/MappedConvolutions)
- Spatial Group-wise Enhance: Enhancing Semantic Feature Learning in Convolutional Networks. arXiv201905 [[Code]](https://github.com/implus/PytorchInsight) [[Note]](https://zhuanlan.zhihu.com/p/66928045)

- [2019ICCV] ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks [[Code]](https://github.com/DingXiaoH/ACNet)
- [2019CVPRW] Convolutions on Spherical Images
- [2017ICML] **Warped Convolutions**: Efficient Invariance to Spatial Transformations
- **Attention module** 
- ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks, arXiv201910 [[Code]](https://github.com/BangguWu/ECANet) [[Chinese]](https://blog.csdn.net/tjut_zdd/article/details/102600401)
- [2020ICLR] On the Relationship between Self-Attention and Convolutional Layers [[Proj]](http://jbcordonnier.com/posts/attention-cnn/) [[Code]](https://github.com/epfml/attention-cnn) [[Intro]](https://zhuanlan.zhihu.com/p/104026923)
- [2019TIP] Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition [[Code]](https://github.com/kaiwang960112/Challenge-condition-FER-dataset)
- [2017CVPR] SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning

**Backbone**

- Comb Convolution for Efficient Convolutional Architecture. [arXiv201911](https://arxiv.org/abs/1911.00387)
- [2019ICML] **EfficientNet**: Rethinking Model Scaling for Convolutional Neural Networks [[Code]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- [2019ICCVW] GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond [[Code]](https://github.com/xvjiarui/GCNet)
- [2018CSVT] [**RoR**] Residual Networks of Residual Networks: Multilevel Residual Networks
- [2018CVPR] [**SENet**] Squeeze-and-excitation networks
- [2017ICLR] FractalNet: Ultra-Deep Neural Networks without Residuals
- [2017CVPR] [PyramidNet] Deep Pyramidal Residual Networks
- [2017CVPR] [**DenseNet**] Densely Connected Convolutional Networks
- [2017CVPR] [**ResNeXt**] Aggregated Residual Transformations for Deep Neural Networks
- [2017CVPR] **Xception**: Deep Learning with Depthwise Separable Convolutions
- [2017CVPR] PolyNet: A Pursuit of Structural Diversity in Very Deep Networks [[Slides]](http://image-net.org/challenges/talks/2016/polynet_talk.pdf)
- [2017AAAI] **Inception-v4**, Inception-ResNet and the Impact of Residual Connections on Learning
- [2016CVPR] [**ResNet**] Deep Residual Learning for Image Recognition [[Note1]](https://mp.weixin.qq.com/s/5gFpyZBUzUz0Y_culZGTFQ) [[Note2]](https://mp.weixin.qq.com/s/bofPG1Vm0RvnH2KaLXYw-Q)
- [2016CVPR] [**Inception-v3**] Rethinking the Inception Architecture for Computer Vision
- [2016ECCV] Good Practices for Deep Feature Fusion
- [2016ECCV] Deep Networks with Stochastic Depth
- [2016ECCV] [**Identity ResNet**] Identity Mappings in Deep Residual Networks [Over 1000 Layers ]
- [2016ICLRW] ResNet in ResNet: Generalizing Residual Architectures
- [2015NIPS] [**STN**] Spatial Transformer Networks
- [2015ICML] [**BN-Inception /Inception-v2**] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- [2015CVPR] [**GoogLeNet/Inception-v1**] Going Deeper with Convolutions
- [2015ICLR] [**VGGNet**] Very Deep Convolutional Networks for Large-Scale Image Recognition
- [2014ICLR] [**NIN**] Network in Network
- [2014ECCV] [**ZFNet**] Visualizing and Understanding Convolutional Networks
- [2014MMM] [**CaffeNet**] Caffe: Convolutional Architecture for Fast Feature Embedding
- [2012NIPS] [**AlexNet**] Imagenet classification with deep convolutional neural networks
- [1998ProcIEEE] [**LeNet**] Gradient-Based Learning Applied to Document Recognition [[LeNet Notes]](https://medium.com/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17)

**Light-weightCNN**

- > [Blog] [Introduction of light-weight CNN](https://zhuanlan.zhihu.com/p/64400678) 
  >
  > [Blog] [Lightweight convolutional neural network: SqueezeNet、MobileNet、ShuffleNet、Xception](https://mp.weixin.qq.com/s/4ROE2hkHHBm11_z_zqM7bQ)

- SeesawNet: Convolution Neural Network With Uneven Group Convolution. [arXiv201912](https://arxiv.org/abs/1905.03672) [[Code]](https://github.com/cvtower/SeesawNet_pytorch)

- HGC: Hierarchical Group Convolution for Highly Efficient Neural Network, arXiv201906

- [2019CVPR] **ESPNetv2**: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network [[Code]](https://github.com/sacmehta/ESPNetv2)

  [2018ECCV] **ESPNet**: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

- [2019CVPRW] Depth-wise Decomposition for Accelerating Separable Convolutions in Efficient Convolutional Neural Networks

- [2019BMVC] MixNet: Mixed Depthwise Convolutional Kernels [[Code]](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) [[Notes]](https://mp.weixin.qq.com/s/U3hP5wJloqE_bJyT__bqvw)

- [2018NIPS] ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions [[Code]](https://github.com/HongyangGao/ChannelNets)

- [2018NIPS] Learning Versatile Filters for Efficient Convolutional Neural Networks [[Code]](https://github.com/huawei-noah/Versatile-Filters)

- [2018BMVC] **IGCV3**: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks [[Code]](https://github.com/homles11/IGCV3) [[Pytorch]](https://github.com/xxradon/IGCV3-pytorch) 

  [2018CVPR] **IGCV2**: Interleaved Structured Sparse Convolutional Neural Networks

  [2017ICCV] [**IGVC1**] Interleaved Group Convolutions for Deep Neural Networks

- MobileNet Series: 

  [Blog] [Introduction for MobileNet and Its Variants](https://mp.weixin.qq.com/s/l8KGdtJo5t0Ze-J7ZL72kQ) 

  [2019ICCV] Searching for MobileNetV3. [[Note]](https://mp.weixin.qq.com/s/Nc84eIk_PhRZWK6dI7BflA)

  [2018CVPR] MobileNetV2: Inverted Residuals and Linear Bottlenecks. [[Note]](zhuanlan.zhihu.com/c_1113861154916601856)

  [2017] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. [arXiv201704](https://arxiv.org/abs/1704.04861)

- ShuffleNet Series [[Note]](https://mp.weixin.qq.com/s/2JSy5FiKDAkVkEl2o1v-aQ) 

  [Code] [ShuffleNet Series by Megvii]((https://github.com/megvii-model/ShuffleNet-Series)): ShuffleNetV1, V2/V2+/V2.Large/V2.ExLarge, OneShot, DetNAS

  [2018ECCV] ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

  [2018CVPR] ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices



### Interpretation

> [Blog] [深度神经网络可解释性方法汇总(附TF代码实现)](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247492729&idx=3&sn=c55dce17e892d73f49a5bbfd955bf899&chksm=f9a196f6ced61fe05f3d085573a8c454d7bd8f49150b84c7e5c8c8c81951cb492a24dbb8358c&mpshare=1&scene=23&srcid=1125qeunZJ3ofeeLVC6rSytw&sharer_sharetime=1574683549503&sharer_shareid=d3d8827bce826478944c5e3a9f67ed4b%23rd)

- Analysis of Explainers of Black Box Deep Neural Networks for Computer Vision: A Survey. [arXiv2019.11](https://arxiv.org/abs/1911.12116)
- [2019NIPS] Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent [[Code]](https://github.com/google/neural-tangents) [[Note]](https://mp.weixin.qq.com/s/lcgnnXMUO8C3oLUPmxthjQ)
- [2019NIPS] Weight Agnostic Neural Networks (spotlight). [[Proj]](https://weightagnostic.github.io/) [[Note]](https://mp.weixin.qq.com/s/NezjvQPp6RZRy3eo_rCj9Q)
- [2018AAAI] Interpreting CNN Knowledge via An Explanatory Graph
- [2018Acces] Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)



## Others

- RetinaFace: Single-stage Dense Face Localisation in the Wild. [arXiv201905](https://arxiv.org/abs/1905.00641) [[Code-MXNet]](https://github.com/deepinsight/insightface/tree/master/RetinaFace) [[Code-TF]](https://github.com/OFRIN/Tensorflow_RetinaFace)
- [2019CVPR] Group Sampling for Scale Invariant Face Detection [[Note]](https://mp.weixin.qq.com/s/MJO73yrtVL--z1fBjkajHg)
- [2019ICCV] Learning to Paint with Model-based Deep Reinforcement Learning [[Code]](https://github.com/hzwer/ICCV2019-LearningToPaint) [[Note]](https://zhuanlan.zhihu.com/p/61761901)

- [2019ICCV] Fashion++: Minimal Edits for Outfit Improvement (FAIR) [[Proj]](http://vision.cs.utexas.edu/projects/FashionPlus/) [[Code]](https://github.com/facebookresearch/FashionPlus)

- [2019ICCV] SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition [[Code&Dataset]](https://github.com/princeton-vl/SpatialSense)
- [2018BMVC] Learning Geo-Temporal Image Features [[Proj]](http://cs.uky.edu/~ted/research/whenwhere/)



**AI+Music**

- [2018ISMIR] MIDI-VAE: Modeling Dynamics and Instrumentation of Music with Applications to Style Transfer [[Code]](https://github.com/brunnergino/MIDI-VAE)
- Music continue: [MuseNet](openai.com/blog/musenet) [Bach-AI-Music-Google](https://www.google.com/doodles/celebrating-johann-sebastian-bach) [Generating Piano Music with Transformer by Google](https://colab.research.google.com/notebooks/magenta/piano_transformer/piano_transformer.ipynb#scrollTo=QI5g-x4foZls)
- On the Measure of Intelligence. [arXiv201911](https://arxiv.org/abs/1911.01547) [[Intro]](https://mp.weixin.qq.com/s?__biz=MzI5NTIxNTg0OA==&mid=2247498902&idx=1&sn=cd4a28b658d8f14d4b2b327f873bb1f5&chksm=ec544b11db23c2079f882e8954ee6ce1feebfcab1bdeef7d2fad4a8ed75bb274129a990f89de&scene=21#wechat_redirect)



**Unsupervised Learning**

- A Simple Framework for Contrastive Learning of Visual Representations. [arXive202002](https://arxiv.org/abs/2002.05709)

