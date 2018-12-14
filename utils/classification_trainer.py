from utils.trainer import Trainer
from utils.helper import Save_Handle
import os
import sys
import time
import torch
from torchvision import transforms
from torch import optim
from torch import nn
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import models
import datasets


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ClassTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""

        """setting contex"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            raise Exception("gpu is not available")

        Dataset = getattr(datasets, args.data_name)
        self.datasets = {x: Dataset(os.path.join(args.data_dir, x), data_transforms[x])
                         for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x=='train' else False),
                                                           num_workers=args.num_workers, pin_memory=True)
                            for x in ['train', 'val']}

        self.model = getattr(models, args.model_name)(Dataset.num_classes, args.pretrained)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.strip(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")
        #
        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=args.device))
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        """training process"""
        args = self.args

        step = 0
        best_acc = 0.0
        step_loss = 0.0
        step_acc = 0
        step_count = 0
        step_start = time.time()

        save_list = Save_Handle(max_num=args.max_model_num)

        for epoch in range(self.start_epoch, args.max_epoch):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_start = time.time()
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                epoch_loss = 0.0
                epoch_acc = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        temp_correct = torch.sum(preds == labels.data)
                        temp_loss_sum = loss.item() * inputs.size(0)

                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            step_loss += temp_loss_sum
                            step_acc += temp_correct
                            step_count += inputs.size(0)

                            if step % args.display_step == 0:
                                step_loss = step_loss / step_count
                                step_acc = step_acc.double() / step_count
                                temp_time = time.time()
                                train_elap = temp_time - step_start
                                step_start = temp_time
                                batch_elap = train_elap / args.display_step if step != 0 else train_elap
                                samples_per_s = 1.0*step_count/train_elap
                                logging.info('Step {} Epoch {}, Train Loss: {:.4f} Train Acc: {:.4f}, '
                                             '{:.1f} examples/sec {:.2f} sec/batch'
                                             .format(step, epoch, step_loss, step_acc, samples_per_s, batch_elap))
                                step_loss = 0.0
                                step_acc = 0
                                step_count = 0
                            step += 1

                    # statistics
                    epoch_loss += temp_loss_sum
                    epoch_acc += temp_correct

                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc.double() / len(self.dataloaders[phase].dataset)

                logging.info('Epoch {} {}, Loss: {:.4f} Acc: {:.4f}, Cost {:.1f} sec'
                             .format(epoch, phase, epoch_loss, epoch_acc, time.time()-epoch_start))


                model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'model_state_dict': model_state_dic
                }, save_path)
                save_list.append(save_path)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))

