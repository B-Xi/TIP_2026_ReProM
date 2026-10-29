# Recurrent Prompting With MambaFormer for Domain Generalized Few-Shot Hyperspectral Image Classification

This is the official code of ReProM. 
Paper web page: [Recurrent Prompting With MambaFormer for Domain Generalized Few-Shot Hyperspectral Image Classification](https://ieeexplore.ieee.org/abstract/document/11739622).

## Abstract
Cross-domain few-shot learning enables the transfer of knowledge from the source domain (SD) to the target domain (TD) with a limited number of annotated samples, proving highly effective in hyperspectral image (HSI) classification. Most existing methods involve iterative training between the SD and TD to achieve few-shot domain adaptation. However, the domain gap still compromises the overall performance of the model on the TD. Additionally, compared to the transformer criticized for speed and memory consumption, the Mamba framework has emerged as a promising alternative with comparable long-distance modeling capabilities while being simplified to linear computational complexity. To this end, we propose a recurrent prompting approach using a Mamba-transformer network (MambaFormer) for domain generalized few-shot HSI classification, referred to as ReProM. Specifically, we develop a novel domain-switch meta-learning strategy (DSMLS) that incorporates cross-domain scenarios into the training process through a “fast switching” manner. This strategy includes a domain-specific prompter designed to prevent the model from overfitting to the current domain and to mitigate catastrophic forgetting of knowledge from previous domains, thereby enhancing the generalization ability of the model. To lower the computational demands of the vanilla transformer, the devised MambaFormer integrates the bidirectional Mamba (BiMamba) and transformer architectures to effectively capture essential spectral-spatial information. Comprehensive experiments conducted on three benchmark HSI datasets reveal that the proposed ReProM achieves better generalization than state-of-the-art methods.

## Citation
```
@ARTICLE{Xi_TIP2026_ReProM,
  author={Xi, Bobo and Zhang, Yun and Li, Jiaojiao and Li, Yunsong and Li, Zan and Chanussot, Jocelyn},
  journal={IEEE Transactions on Image Processing}, 
  title={Recurrent Prompting With MambaFormer for Domain Generalized Few-Shot Hyperspectral Image Classification}, 
  year={2026},
  volume={35},
  number={},
  pages={11002-11015},
  keywords={Modeling;Training;Transformers;Timing;Hyperspectral imaging;Image classification;Few shot learning;Educational institutions;Architecture;Computer architecture;Few-shot learning;domain generalized;catastrophic forgetting;Mamba},
  doi={10.1109/TIP.2026.3742663}}
```

## Requirements
* PyTorch version: 2.4.1
* Python version: 3.9.20
* numpy: 2.0.2
* mamba_ssm: 1.2.0.post1
 
## Usage
* For KSC dataset, you can run
* Stage 1 `python mamba_myproto.py`
* Stage 2, Stage 3 `python finetuning_KSC.py`
