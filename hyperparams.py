# datasets
train_set = "train_wiki"
val_set = "val_pubmed"
test_set = "val_pubmed"
adv_set = "pubmed_unsupervised"

# adv_set = None
# val_set = "val_wiki"
# test_set = "val_wiki"

# adv_set = "remain_unsup"

# N-way K-shot
batch_size = 4
trainN = 10
N = 5
K = 1
Q = 5

# model params
model = "relation"
encoder = "cnn"
hidden_size = 230
dropout = 0.1
max_length = 128

coef = 0.25
tau = 0.5
anneal_step = 6000
anneal_mode = "cosine"
n_clusters = 10
cluster = False
pseudo_pth = "train_wiki_and_pseudo_pubmed"
feature_pth = "unlabel_features"

# train params
train_iter = 10000
val_iter = 1000
val_step = 1000
test_iter = 10000

optim = "sgd"
lr = 1e-1
lr_step_size=20000
weight_decay = 1e-5
adv_dis_lr=1e-1
adv_enc_lr=1e-1
warmup_step=300

seed = None # 1023, 42

# save and load
load_ckpt = None
save_ckpt = None
only_test = False

# others
fp16 = False
pair = False
na_rate = 0
grad_iter = 1
