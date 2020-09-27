Inductive Unsupervised Domain Adaptation for Few-Shot Classification via Clustering
===

This code is for ECML-PKDD 2020 paper "Inductive Unsupervised Domain Adaptation for Few-Shot Classification via Clustering".

Requirements
---

python=3.7
pytorch=1.1.0
transformers=2.7.0

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
Please evaluate the performance of your model in the FewRel 2.0 official website [https://thunlp.github.io/2/fewrel2_da.html](https://thunlp.github.io/2/fewrel2_da.html).

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

