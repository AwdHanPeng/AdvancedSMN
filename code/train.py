from model import Model, Config
import argparse
import torch
import torch.optim as optim
from dataset import TrainDataset, ValidDataset
from utils import *
import pickle

'''
    此文件定义了模型的训练过程 以及数据padding操作和众多评测指标的实现
'''
'''
    Adam algorithm
    initial learning rate is 0.001
    the parameters of Adam, β1 and β2 are 0.9 and 0.999 respectively
    employed early-stopping as a regularization strategy
    Models were trained in mini batches with a batch size of 200, and the maximum
    utterance length is 50
    We padded zeros if the number
    of utterances in a context is less than 10, otherwise
    we kept the last 10 utterances
'''

parser = argparse.ArgumentParser(description='Train')

parser.add_argument('--batch-size', type=int, default=500,
                    help='input batch size for training (default: 500)')
parser.add_argument('--valid-batch-size', type=int, default=1000,
                    help='input batch size for validation (default: 1000)')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta 1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta 2 (default: 0.999)')
parser.add_argument('--no-cuda', default=0, type=int,
                    help='disables CUDA training')
parser.add_argument('--save-model', default=1, type=int,
                    help='For Saving the current Model')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--advance', default=0, type=int,
                    help='whether to use a new matrix (A1_matrix) which is not used in paper ')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_root = ['utterance', 'correct_response', 'error_response']
train_root_path = ['../Data/train/' + i + '.pkl' for i in train_root]
valid_root = ['utterance', 'responses', 'index']
valid_root_path = ['../Data/valid/' + i + '.pkl' for i in valid_root]

train_loader = torch.utils.data.DataLoader(
    TrainDataset(root=train_root_path, transforms=[padding_sentence, padding_utterance]),
    batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(
    ValidDataset(root=valid_root_path, transforms=[padding_sentence, padding_utterance]),
    batch_size=args.valid_batch_size, shuffle=False, **kwargs)

'''
    nll_loss:
        input:(N,C) where C = number of classes
        target:(N) where each value is 0≤ target ≤C−1,
'''
if __name__ == '__main__':
    history_list = []
    config = Config(advance=args.advance)
    model = Model(config)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    model = model.to(device)
    best_p1 = 0
    best_epo = 1
    last = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        MAP, R10_1, R10_2, R10_5 = valid(args, model, device, valid_loader)
        history_list.append({
            'epoch': epoch,
            'MRR': MAP,
            'MAP': MAP,
            'P_1': R10_1,
            'R10_1': R10_1,
            'R10_2': R10_2,
            'R10_5': R10_5,
        })
        if args.save_model and R10_1 > best_p1:
            if args.advance:
                torch.save(model.state_dict(), "../model/Advance{}.pt".format(epoch))
            else:
                torch.save(model.state_dict(), "../model/Primitive{}.pt".format(epoch))
            print('save model when epoch = {} as {}{}.pt'.format(epoch, 'Advance' if args.advance else 'Primitive',
                                                                 epoch))
            best_p1 = R10_1
            best_epo = epoch
        if R10_1 < best_p1:
            last += 1
        if last >= 2:
            break
    print('best p@1 = {} at epoch {}'.format(best_p1, best_epo))
    with open('../model/history_{}.pkl'.format('Advance' if args.advance else 'Primitive'), 'wb') as f:
        pickle.dump(history_list, f)
