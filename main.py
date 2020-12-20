import os
import time
import datetime

import torch

from modeling import build_model
from utils import build_train_loader, build_optimizer
from utils import MetricLogger, SmoothedValue
from utils.arguments import parse_args


def train_one_epoch(model, optimizer,  lr_scheduler, data_loader, epoch, print_freq, checkpoint_fn=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'batch/s', SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)

    for step, batched_inputs in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()
        loss = model(batched_inputs)

        if checkpoint_fn is not None and np.random.random() < 0.005:
            checkpoint_fn()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['batch/s'].update((time.time() - start_time))
        lr_scheduler.step()

    if checkpoint_fn is not None:
        checkpoint_fn()


def main():
    # parse args
    args = parse_args()

    # build data_loader
    file_path = args.file_path
    data_loader = build_train_loader(file_path)

    device = torch.device("cpu")
    model = build_model().to(device)

    optimizer = build_optimizer(model, lr=args.lr)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

    def save_model_checkpoint():
        if args.output_dir:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

    print("Start training")
    start_time = time.time()
    import ipdb; ipdb.set_trace()
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, lr_scheduler, data_loader,
                        epoch, args.print_freq, checkpoint_fn=save_model_checkpoint)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
