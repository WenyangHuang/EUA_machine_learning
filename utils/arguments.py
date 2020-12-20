import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--file-path', type=str,
                        help='file path', default="data/constrained.csv")
    parser.add_argument('--batch-szie', default=8, help='batch size', type=int)
    parser.add_argument('--epochs', default=25, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+',
                        default=[20, 30, 40], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.3, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0,
                        type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=10,
                        type=int, help='print frequency')
    parser.add_argument('--output-dir', default='auto',
                        help='path where to save')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='start epoch')

    args = parser.parse_args()
    return args
