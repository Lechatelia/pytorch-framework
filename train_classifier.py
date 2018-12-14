from utils.classification_trainer import ClassTrainer
import argparse
import os
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')

    parser.add_argument('--model-name', default='resnet50', help='the name of the model')
    parser.add_argument('--data-name', default='Shoes', help='the name of the model')
    parser.add_argument('--data-dir', default='/home/teddy/shoes', help='training set directory')
    parser.add_argument('--save-dir', default='/home/teddy/shoes/model', help='directory to save model.')
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to load pretrained model')
    parser.add_argument('--device', default='2,3', help='assign device')
    parser.add_argument('--batch-size', type=int, default=200, help='input batch size.')
    parser.add_argument('--num-workers', type=int, default=8, help='the num of training process')

    parser.add_argument('--opt', default='sgd', help='the optimizer method')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum of gradient')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr-scheduler', default='fix', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma multiplied at each step')
    parser.add_argument('--steps', default='30,45', help='the learning rate decay steps')

    parser.add_argument('--resume', default='', help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=20, help='most recent models num to save ')
    parser.add_argument('--max-epoch', type=int, default=100, help='max training epoch')
    parser.add_argument('--display-step', type=int, default=100,
                        help='the num of steps to log training information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = ClassTrainer(args)
    trainer.setup()
    trainer.train()
