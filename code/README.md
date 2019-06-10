command:
    CUDA_VISIBLE_DEVICES=2 python train.py -h
    source activate torch
    nvidia-smi
    tar zxvf filename.tar
    tar czvf filename.tar dirname

total_words = 218593 = pad unk +num of words
max_num_utterance = 10
max_sentence_len = 20
GRU1_hidden_size = 200
GRU2_hidden_size = 50
v_length = 50

1) nn.Parameter 不然你的模型叶变量是不会更新的 而且保存的时候模型也没办法存
2) 学习率很重要 如果不收敛 lr->0.0001
3) max_sentence_len,max_num_utterance 需要根据统计情况确定
不能太大 不能太小 最好是两倍 这个非常重要 太大了padding的0太多 有可能概率全一样了
4) padding 在transform里边做 如果要做mask，那么需要把长度传进去，这样就可以在模型里边mask了
5) batch_size 不太重要 但是也需要考虑
6) embedding 如果要预处理 一定要考虑好构造矩阵的方案 节省时间
7) 爱因斯坦求和约定很nice
8) dataset 里边返回item的时候要做torch.tensor
9) util的使用 模块化
10) argparse很好用 但是一定要注意bool值是没办法传的 用01 int挺好的
11) 函数中要小心可能会把实参改动掉了
12) 双向的GRU应该会好点 utterance和response的utterance用成一样的可能会好点 这样可以将他们放到一个状态空间里
但是可能train不出来
13) permute之后就不能view了
14) 同时遍历两个list用zip函数
15) layer的初始化 卷积核用kaiming_normal_ GRU用orthogonal_ linear用xavier_uniform_
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        w = (param.data for name, param in m.named_parameters() if 'weight' in name)
        for k in w:
            nn.init.orthogonal_(k)
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)