from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised, get_loader_all_unsupervised
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder
import models
from models.proto import Proto
from models.relation import Relation
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
from models.pair import Pair
from models.d import Discriminator
from models.entropy import SimilarityEntropy

import hyperparams as hp

import torch
from torch import optim, nn
import numpy as np

import sys
import json
import argparse
import os
import time
import random
import pdb

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train', default=hp.train_set,
            help='train file')
    parser.add_argument('--val', default=hp.val_set,
            help='val file')
    parser.add_argument('--test', default=hp.test_set,
            help='test file')
    parser.add_argument('--adv', default=hp.adv_set,
            help='adv file')
   
    parser.add_argument('--batch_size', default=hp.batch_size, type=int,
            help='batch size')
    parser.add_argument('--trainN', default=hp.trainN, type=int,
            help='N in train')
    parser.add_argument('--N', default=hp.N, type=int,
            help='N way')
    parser.add_argument('--K', default=hp.K, type=int,
            help='K shot')
    parser.add_argument('--Q', default=hp.Q, type=int,
            help='Num of query per class')
    
    parser.add_argument('--model', default=hp.model,
            help='model name')
    parser.add_argument('--encoder', default=hp.encoder,
            help='encoder: cnn or bert')
    parser.add_argument('--hidden_size', default=hp.hidden_size, type=int,
           help='hidden size')
    parser.add_argument('--dropout', default=hp.dropout, type=float,
           help='dropout rate')
    parser.add_argument('--max_length', default=hp.max_length, type=int,
           help='max length')

    parser.add_argument('--coef', default=hp.coef, type=float,
           help='coef')
    parser.add_argument('--tau', default=hp.tau, type=float,
           help='tau')
    parser.add_argument('--anneal_step', default=hp.anneal_step, type=int,
           help='anneal step')
    parser.add_argument('--anneal_mode', default=hp.anneal_mode,
           help='anneal mode')
    parser.add_argument('--n_clusters', default=hp.n_clusters, type=int,
           help='num of clusters')
    parser.add_argument('--cluster', default=hp.cluster, action="store_true",
           help='cluster')
    parser.add_argument('--pseudo_pth', default=hp.pseudo_pth)
    parser.add_argument('--feature_pth', default=hp.feature_pth)

    parser.add_argument('--train_iter', default=hp.train_iter, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=hp.val_iter, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=hp.test_iter, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=hp.val_step, type=int,
           help='val after training how many iters')
    
    parser.add_argument('--optim', default=hp.optim,
           help='sgd / adam / adamw')
    parser.add_argument('--lr', default=hp.lr, type=float,
           help='learning rate')
    parser.add_argument('--lr_step_size', default=hp.lr_step_size, type=int,
           help='learning rate step')
    parser.add_argument('--weight_decay', default=hp.weight_decay, type=float,
           help='weight decay')
    parser.add_argument('--adv_dis_lr', default=hp.adv_dis_lr, type=float,
           help='adv dis lr')
    parser.add_argument('--adv_enc_lr', default=hp.adv_enc_lr, type=float,
           help='adv enc lr')
    parser.add_argument('--warmup_step', default=hp.warmup_step, type=int,
           help='warmup step')
    
    parser.add_argument('--load_ckpt', default=hp.load_ckpt,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=hp.save_ckpt,
           help='save ckpt')
    parser.add_argument('--only_test', action='store_true', default=hp.only_test,
           help='only test')

    parser.add_argument('--seed', default=hp.seed, type=int,
           help='seed')

    parser.add_argument('--na_rate', default=hp.na_rate, type=int,
           help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=hp.grad_iter, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--fp16', action='store_true', default=hp.fp16,
           help='use nvidia apex fp16')
    parser.add_argument('--pair', action='store_true', default=hp.pair,
           help='use pair model')

    opt = parser.parse_args()
    print()
    print(opt)
    print()

    if opt.seed is None:
        opt.seed = round((time.time() * 1e8) % 1e8)
    print(f"Seed: {opt.seed}\n")
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    
    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length,
                hidden_size=opt.hidden_size
                )
    elif encoder_name == 'bert':
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                    './pretrain/bert-base-uncased',
                    max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                    './pretrain/bert-base-uncased',
                    max_length)
    else:
        raise NotImplementedError
    
    if opt.pair:
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader_pair(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    else:
        train_data_loader = get_loader(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        if opt.adv:
           adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv:
        d = Discriminator(opt.hidden_size)
        se = SimilarityEntropy(coef=opt.coef, tau=opt.tau)
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader, adv=opt.adv, d=d, se=se)
    else:
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    prefix = '-'.join([timestamp, model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    
    if model_name == 'proto':
        model = Proto(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N)
    elif model_name == 'snail':
        print("HINT: SNAIL works only in PyTorch 0.3.1")
        model = SNAIL(sentence_encoder, N, K)
    elif model_name == 'metanet':
        model = MetaNet(N, K, sentence_encoder.embedding, max_length)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'relation':
        model = Relation(sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    print(f"Checkpoint: {ckpt}")

    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name == 'bert':
            bert_optim = True
        else:
            bert_optim = False

        framework.train(
                model, prefix,
                batch_size, trainN, N, K, Q,
                train_iter=opt.train_iter,
                val_iter=opt.val_iter,
                val_step=opt.val_step,
                bert_optim=bert_optim,
                pytorch_optim=pytorch_optim,
                learning_rate=opt.lr,
                lr_step_size=opt.lr_step_size,
                weight_decay=opt.weight_decay,
                adv_dis_lr=opt.adv_dis_lr,
                adv_enc_lr=opt.adv_enc_lr,
                warmup_step=opt.warmup_step,
                anneal_step=opt.anneal_step,
                load_ckpt=opt.load_ckpt,
                save_ckpt=ckpt,
                na_rate=opt.na_rate,
                fp16=opt.fp16,
                pair=opt.pair, 
                )
    else:
        ckpt = opt.load_ckpt

    if opt.cluster:
        unlabel_data_loader = get_loader_all_unsupervised(opt.adv, sentence_encoder,
            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        framework.cluster(model, ckpt, unlabel_data_loader, opt.n_clusters, opt.pseudo_pth, opt.feature_pth)
        print()
        print("cluster over")
        print()

    print(opt)
    for n in [5, 10]:
        for k in [1, 5]:
            test_data_loader = get_loader(opt.test, sentence_encoder, N=n, K=k, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
            framework.test_data_loader = test_data_loader
            acc = framework.eval(model, batch_size, n, k, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
            print(f"{n}-way-{k}-shot accuracy : {acc*100:.2f}")

if __name__ == "__main__":
    main()
