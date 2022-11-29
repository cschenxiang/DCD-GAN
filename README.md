<<<<<<< HEAD
# Unpaired Deep Image Deraining Using Dual Contrastive Learning

<hr />

> **Abstract:** *Learning single image deraining (SID) networks from an unpaired set of clean and rainy images is practical and valuable as acquiring paired real-world data is almost infeasible. However, without the paired data as the supervision, learning a SID network is challenging. Moreover, simply using existing unpaired learning methods (e.g., unpaired adversarial learning and cycle-consistency constraints) in the SID task is insufficient to learn the underlying relationship from rainy inputs to clean outputs as there exists significant domain gap between the rainy and clean images. In this paper, we develop an effective unpaired SID adversarial framework which explores mutual properties of the unpaired exemplars by a dual contrastive learning manner in a deep feature space, named as DCD-GAN. The proposed method mainly consists of two cooperative branches: Bidirectional Translation Branch (BTB) and Contrastive Guidance Branch (CGB). Specifically, BTB exploits full advantage of the circulatory architecture of adversarial consistency to generate abundant exemplar pairs and excavates latent feature distributions between two domains by equipping it with bidirectional mapping. Simultaneously, CGB implicitly constrains the embeddings of different exemplars in the deep feature space by encouraging the similar feature distributions closer while pushing the dissimilar further away, in order to better facilitate rain removal and help image restoration. Extensive experiments demonstrate that our method performs favorably against existing unpaired deraining approaches on both synthetic and real-world datasets, and generates comparable results against several fully-supervised or semi-supervised models.* 
=======
# Unpaired Deep Image Dehazing Using Contrastive Disentanglement Learning

<hr />

> **Abstract:** *We offer a practical unpaired learning based image dehazing network from an unpaired set of clear and hazy images. This paper provides a new perspective to treat image dehazing as a two-class separated factor disentanglement task, i.e., the task-relevant factor of clear image reconstruction and the task-irrelevant factor of haze-relevant distribution. To achieve the disentanglement of these two-class factors in deep feature space, contrastive learning is introduced into a CycleGAN framework to learn disentangled representations by guiding the generated images to be associated with latent factors. With such formulation, the proposed contrastive disentangled dehazing method (CDD-GAN) employs negative generators to cooperate with the encoder network to update alternately, so as to produce a queue of challenging negative adversaries. Then these negative adversaries are trained end-to-end together with the backbone representation network to enhance the discriminative information and promote factor disentanglement performance by maximizing the adversarial contrastive loss. During the training, we further show that hard negative examples can suppress the task-irrelevant factors and unpaired clear exemples can enhance the task-relevant factors, in order to better facilitate haze removal and help image restoration. Extensive experiments on both synthetic and real-world datasets demonstrate that our method performs favorably against existing unpaired dehazing baselines.* 
>>>>>>> c637665b88f44ca962fa9521284adeb6fab6e28b
<hr />

## Network Architecture

<img src = "figure/network.png"> 

## Citation
If you are interested in this work, please consider citing:

    @inproceedings{chen2022unpaired,
        title={Unpaired Deep Image Deraining Using Dual Contrastive Learning}, 
        author={Chen, Xiang and Pan, Jinshan and Jiang, Kui and Li, Yufeng and Huang, Yufeng and Kong, Caihua and Dai, Longgang and Fan, Zhentao},
        booktitle={CVPR},
        year={2022}
    }

## Acknowledgment
<<<<<<< HEAD
This code is based on the [DCLGAN](https://github.com/JunlinHan/DCLGAN). Thanks for sharing !
=======
This code is based on the [NEGCUT](https://github.com/WeilunWang/NEGCUT). Thanks for sharing !
>>>>>>> c637665b88f44ca962fa9521284adeb6fab6e28b
