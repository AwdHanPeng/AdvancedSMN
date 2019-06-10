import pickle
import torch
import argparse
from utils import *
from dataset import ValidDataset
from model import Model, Config

'''
    'Advance' token shows this model do not use A1_matrix just consistent with paper description
'''
parser = argparse.ArgumentParser(description='Valid')
parser.add_argument('--valid-batch-size', type=int, default=1000,
                    help='input batch size for validation (default: 1000)')
parser.add_argument('--no-cuda', default=0, type=int,
                    help='disables CUDA training')
parser.add_argument('--valid', default=0, type=int,
                    help='whether to valid model right now')
parser.add_argument('--img', default=0, type=int,
                    help='whether to show history image right now')
parser.add_argument('--advance', default=0, type=int,
                    help='whether to use a new matrix (A1_matrix) which is not used in paper ')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
valid_root = ['utterance', 'responses', 'index']
valid_root_path = ['../Data/valid/' + i + '.pkl' for i in valid_root]
valid_loader = torch.utils.data.DataLoader(
    ValidDataset(root=valid_root_path, transforms=[padding_sentence, padding_utterance]),
    batch_size=args.valid_batch_size, shuffle=False, **kwargs)

if __name__ == '__main__':
    if args.valid:
        config = Config(advance=args.advance)
        model = Model(config)
        model.load_state_dict(
            torch.load('../model/{}.pt'.format(best_advance_model if args.advance else best_primitive_model)))
        model = model.to(device)
        valid(args, model, device, valid_loader)
    if args.img:
        with open('../model/history_{}.pkl'.format('Advance' if args.advance else 'Primitive'), 'rb') as f:
            history_list = pickle.load(f)
        imshow(history_list, title='{}'.format('Advance' if args.advance else 'Primitive'))
