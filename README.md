Inductive Unsupervised Domain Adaptation for Few-Shot Classification via Clustering
===

This code is for ECML-PKDD 2020 paper "Inductive Unsupervised Domain Adaptation for Few-Shot Classification via Clustering".

In this paper, we introduce a model-agnostic framework, DaFeC, to improve domain adaptation performance for few-shot classification via clustering. We first build a representation extractor to derive features for unlabeled data from the target domain (no test data is necessary) and then group them with a cluster miner. The generated pseudo-labeled data and the labeled source-domain data are used as supervision to update the parameters of the few-shot classifier. In order to derive high-quality pseudo labels, we propose a Clustering Promotion Mechanism, to learn better features for the target domain via Similarity Entropy Minimization and Adversarial Distribution Alignment, which are combined with a Cosine Annealing Strategy. Experiments are performed on the FewRel 2.0 dataset. Our approach outperforms previous work with absolute gains (in classification accuracy) of 4.95%, 9.55%, 3.99% and 11.62%, respectively, under four few-shot settings.

You can find the paper [here](https://arxiv.org/abs/2006.12816).

Requirements
---

Python=3.7

PyTorch=1.1.0

CUDA=9.0

Transformers=2.7.0

Preparation
---

The training and dev dataset have been included in the `./data` directory and the test set is not public. You should evaluate your models in the [offical website](https://thunlp.github.io/2/fewrel2_da.html).

Due to the large size, the pre-trained Glove files and BERT pretrain checkpoint are not included. You can download them in [here](https://cloud.tsinghua.edu.cn/f/58f57bda00eb40be8d10/?dl=1).

Usage
---

1. Training Representation Extractor

```shell
python train_demo.pys --save_ckpt=path_to_your_saved_model
```

2. Generating pseudo-labeled data

```shell
python train_demo.py --load_ckpy=path_to_your_saved_model --cluster
```

3. Training few-shot classifier (BERT-PAIR as the example)

```shell
python train_demo.py --model=pair --pair --encoder=bert --hidden_size=768 --optim=adamw --lr=2e-5 --train=train_wiki_and_pseudo_pubmed --save_ckpt=path_to_your_saved_model
```

4. Evaluation
Please evaluate the performance of your model in the [FewRel 2.0 official website](https://thunlp.github.io/2/fewrel2_da.html).

Citation
---

```
@inproceedings{cong2020DaFeC,
 author = {Cong, Xin and Yu, Bowen and Liu, Tingwen and Cui, Shiyao and  Tang, Hengzhu and Wang, Bin},
 booktitle = {Proc. of ECML-PKDD},
 title = {Inductive Unsupervised Domain Adaptation for Few-Shot Classification via Clustering},
 year = {2020}
}
```

Related Repo
---

The dataset and baselines are adapted from [FewRel](https://github.com/thunlp/FewRel).