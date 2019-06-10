import torch
from dataset import TestDataset
from utils import *
import argparse
import pickle
from model import Model, Config

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--test-batch-size', type=int, default=1000,
                    help='input batch size for validation (default: 1000)')
parser.add_argument('--no-cuda', default=0, type=int,
                    help='disables CUDA training')

parser.add_argument('--advance', default=0, type=int,
                    help='whether to use a new matrix (A1_matrix) which is not used in paper ')
parser.add_argument('--save-index', default=1, type=int,
                    help='whether to save index from model output on test sample')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
test_root = ['utterance', 'responses']
test_root_path = ['../Data/test/' + i + '.pkl' for i in test_root]

if __name__ == "__main__":
    config = Config(advance=args.advance)
    model = Model(config)
    model.load_state_dict(
        torch.load('../model/{}.pt'.format(best_advance_model if args.advance else best_primitive_model)
                   ))
    model = model.to(device)
    test_loader = torch.utils.data.DataLoader(
        TestDataset(root=test_root_path, transforms=[padding_sentence, padding_utterance]),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    index, ranks = test(args, model, device, test_loader)
    if args.save_index:
        rank_out = output(ranks)
        with open('../Data/test/test_ground', 'w', encoding='UTF-8') as f:
            f.writelines(rank_out)
        print('complete to save ranks index')
